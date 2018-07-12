package core.dlm.model

import breeze.stats.distributions._
import breeze.stats.distributions._
import breeze.numerics.exp
import breeze.linalg.{DenseVector, DenseMatrix, diag, svd}
import breeze.stats.mean
import breeze.linalg.svd._
import cats.implicits._
import core.dlm.model._

/**
  * Model a large covariance matrix using a factor structure
  * Note: This framework currently can't handle partially missing observations
  */
object FactorSv {
  /**
    * Factor Stochastic Volatility Parameters for a model with k factors and p time series, 
    * k << p
    * @param v the variance of the measurement error
    * @param beta the factor loading matrix, p x k
    * @param factorParams a vector of length k containing the ISV parameters for the factors 
    */
  case class Parameters(
    v:            Double,
    beta:         DenseMatrix[Double],
    factorParams: Vector[SvParameters]
  )

  /**
    * The state of the Gibbs Sampler
    * @param p the current sample of the fsv parameters
    * @param factors the current sample of the time series of factors, k x n dimensional
    * @param volatility the current sample of the time series of variances, k x n dimensional
    */
  case class State(
    params:     FactorSv.Parameters,
    factors:    Vector[(Double, Option[DenseVector[Double]])],
    volatility: Vector[(Double, DenseVector[Double])]
  )

  /**
    * Build a beta matrix from a prior distribution
    * @param p the dimension of the rows of beta
    * @param k the dimension of the columns of beta
    * @param prior a lazily evaluated prior distribution
    * @return a beta matrix with ones on the leading diagonal, zeros above
    * and draws from the prior distribution elsewhere
    */
  def buildBeta(
    p: Int,
    k: Int,
    prior: => Double): DenseMatrix[Double] = {

    DenseMatrix.tabulate(p, k){ case (i, j) =>
      if (i == j) {
        1.0
      } else if (i > j) {
        prior
      } else {
        0.0
      }
    }
  }

  /**
    * A single step for simulating a factor stochastic volatility model
    * @param dt the time increment between successive realisations 
    * @param t the current time of the realisation
    * @param params the parameters of the stochastic volatility model
    * @param a the current log volatilities of the factors
    * @return 
    */
  def simStep(
    t:      Double,
    params: Parameters)(a: Vector[Double]) = {

    for {
      vt <- MultivariateGaussian(
        DenseVector.zeros[Double](params.beta.rows),
        diag(DenseVector.fill(params.beta.rows)(params.v)))

      fs <- (a zip params.factorParams).
        traverse { case (x, p) => StochasticVolatility.simStep(t, p)(x) }

      f = fs.map { case (t, ft, at) => ft.get }

      a1 = fs.map { case (t, ft, at) => at }

      y = params.beta * DenseVector(f.toArray) + vt
    } yield (Dlm.Data(t, y.map(_.some)), f, a1)
  }

  def simulate(p: Parameters) = {
    val k = p.beta.cols
    val initState = Vector.fill(k)(Gaussian(0.0, 1.0).draw)
    val init = (Dlm.Data(0.0, DenseVector[Option[Double]](None)),
      Vector.fill(k)(0.0), initState)

    MarkovChain(init){ case (d, _, x) => simStep(d.time + 1.0, p)(x) }
  }

  /**
    * Encode partially missing data as totally missing in order to
    * sample the factors
    * @param obs a vector of observations
    * @return
    */
  def encodePartiallyMissing(
    obs: Vector[Dlm.Data]): Vector[(Double, Option[DenseVector[Double]])] = {

    obs.map(d =>
      (d.time, d.observation.data.toVector.sequence.map { x =>
        DenseVector(x.toArray)
      })
    )
  }

  /**
    * Sample the factors from a multivariate gaussian
    *
    * @param beta the current value of the factor loading matrix
    * @param observations the observations of the time series
    * @param volatility the current value of the volatility
    * @param sigmaY the value of the observation variance
    * @return
    */
  def sampleFactors(
    observations: Vector[Dlm.Data],
    p: FactorSv.Parameters,
    volatility: Vector[(Double, DenseVector[Double])]) = {

    val precY = diag(DenseVector.fill(p.beta.rows)(1.0 / p.v))
    val obs = encodePartiallyMissing(observations)
    val beta = p.beta

    // sample factors independently
    val res = for {
      ((time, ys), at) <- obs zip volatility
      fVar = diag(exp(-at._2))
      prec = fVar + (beta.t * precY * beta)
      mean = ys.map(y => prec \ (beta.t * (precY * y)))
      sample = mean map (m => rnorm(m, prec).draw)
    } yield (time, sample)

    Rand.always(res)
  }

  /**
    * Select the ith observation from a vector of data representing a
    * multivariate time series
    * @param y a vector of data
    * @param i the index of the observation to select
    * @return a vector containing the ith observation of a multivariate time series
    */
  def getithObs(y: Vector[Dlm.Data], i: Int): Vector[Option[Double]] = {
    y.map(d => d.observation(i))
  }

  /**
    * Sum the product of f and y
    * @param facs the latent factors
    * @param obs the observations of the ith time series
    * @return the squared sum of the product of factors and observations
    */
  def sumFactorsObservations(
    facs: Vector[DenseVector[Double]],
    obs:  Vector[Option[Double]]): DenseVector[Double] = {

    (obs zip facs).
      map {
        case (Some(y), f) => f * y
        case  (None, f) => DenseVector.zeros[Double](f.size)
      }.
      reduce(_ + _)
  }

  /**
    * Generate a random draw from the multivariate normal distribution
    * Using the precision matrix
    */
  def rnorm(
    mean: DenseVector[Double],
    prec: DenseMatrix[Double]) = new Rand[DenseVector[Double]] {

    def draw = {
      val z = DenseVector.rand(mean.size, Gaussian(0.0, 1.0))
      val SVD(u, d, vt) = svd(prec)
      val dInv = d.map(1.0 / _)

      mean + (vt.t * diag(dInv) * z)
    }
  }

  /**
    * The rows, i = 1,...,p of a k-factor model of Beta can be updated with Gibbs step
    * The prior specification is b_ij ~ N(0, C0) i > j, b_ii = 1, b_ij = 0 for j > i
    * This is a helper function for sampleBeta
    * @param prior a multivariate normal prior distribution for each column
    * @param factors the current value of latent factors
    * @param sigma the variance of the measurement error
    * @param i the row number
    * @param k the total number of factors in the model
    * @return the full conditional distribution of the
    */
  def sampleBetaRow(
    prior:        Gaussian,
    factors:      Vector[DenseVector[Double]],
    observations: Vector[Dlm.Data],
    sigma:        Double,
    i:            Int,
    k:            Int) = {

    if (i < k) {
      // take factors up to f value i - 1
      val fsi = factors.map(x => x(0 until i))

      // take the ith time series observation
      val obs: Vector[Option[Double]] = getithObs(observations, i)

      val id = DenseMatrix.eye[Double](i)
      val sumFactors = fsi.map(fi => fi * fi.t).reduce(_ + _)

      val prec = (1.0 / sigma) * sumFactors + id * prior.variance
      val mean = (1.0 / sigma) * sumFactorsObservations(fsi, obs)

      rnorm(prec \ mean, prec).draw
    } else {
      // take the ith time series observation
      val obs = getithObs(observations, i)

      val id = DenseMatrix.eye[Double](k)
      val sumFactors = factors.map(fi => fi * fi.t).reduce(_ + _)

      val prec = (1.0 / sigma) * sumFactors + id * prior.variance
      val mean = (1.0 / sigma) * sumFactorsObservations(factors, obs)

      rnorm(prec \ mean, prec).draw
    }
  }

  /**
    * Make an empty beta matrix
    * Make a p x k matrix with 1s on the diagonal and zeros elsewhere
    * @param p the rows of the matrix corresponding to the number of time series to model
    * @param k the number of factors
    * @return a p x k matrix with 1s on leading diagonal
    */
  def makeBeta(p: Int, k: Int): DenseMatrix[Double] = {
    DenseMatrix.tabulate(p, k) { case (i, j) =>
      if (i == j) {
        1.0
      } else {
        0.0
      }
    }
  }

  def flattenFactors(fs: Vector[(Double, Option[DenseVector[Double]])]) = {
    fs.map { case (t, fs) => fs.map((t, _)) }.flatten
  }

  /**
    * Sample a value of beta using the function sampleBetaRow
    * @param prior the prior distribution to be used for each element of beta
    * @param observations a time series containing the
    * @param p the dimension of a single observation vector
    * @param k the dimension of the latent-volatilities
    * @return
    */
  def sampleBeta(
    prior:  Gaussian,
    ys:     Vector[Dlm.Data],
    p:      Int,
    k:      Int,
    fs:     Vector[(Double, Option[DenseVector[Double]])],
    params: Parameters
  ) = {

    val newbeta = makeBeta(p, k)
    val fls = flattenFactors(fs.sortBy(_._1)).map(_._2)

    (1 until p).foreach { i =>
      if (i < k) {
        newbeta(i, 0 until i).t := sampleBetaRow(prior, fls, ys, params.v, i, k)
      } else {
        newbeta(i, ::).t := sampleBetaRow(prior, fls, ys, params.v, i, k)
      }}

    Rand.always(newbeta)
  }

  /**
    * Extract a single state from a vector of states
    * @param s the combined state
    * @param i the position of the state to extract
    * @return the extracted state
    */
  def extractState(
    vs: Vector[(Double, DenseVector[Double])],
    i: Int): Vector[(Double, Double)] =
    vs.map { case (t, x) => (t, x(i)) }


  def combineStates(s: Vector[Vector[(Double, Double)]]): Vector[(Double, DenseVector[Double])] = {
    s.transpose.map(x => (x.head._1, DenseVector(x.map(_._2).toArray)))
  }

  /**
    * Extract the ith factor from a multivariate vector of factors
    */
  def extractFactors(
    fs: Vector[(Double, Option[DenseVector[Double]])],
    i:  Int): Vector[(Double, Option[Double])] = {

    fs map { case (t, fo) =>
      (t, fo map (f => f(i)))
    }
  }

  /**
    * Sample each of the factor states and parameters in turn
    * @param priorMu a Normal prior for the mean of the AR(1) process
    * @param priorSigmaEta an inverse Gamma prior for the variance of the AR(1) process
    * @param piorPhi a Beta prior on the mean reversion parameter phi
    * @param p an integer specifying the dimension of the observations
    * @param s the current state of the MCMC algorithm
    * @return 
    */
  def sampleVolatilityParams(
    priorMu:       Gaussian,
    priorSigmaEta: InverseGamma,
    priorPhi:      Beta,
    p:             Int)(s: State) = {

    val k = s.params.beta.cols

    val res = for {
      i <- Vector.range(0, k).par
      thisState = extractState(s.volatility, i)
      theseParameters = s.params.factorParams(i)
      factorState = StochasticVolatility.State(theseParameters, thisState)
      theseFactors = extractFactors(s.factors, i)
      factor = StochasticVolatility.stepArSvd(priorSigmaEta,
        priorPhi, priorMu, theseFactors)(factorState)
    } yield factor

    for {
      res <- res.seq.sequence
      params = res.map(_.params)
      state = res.map(_.alphas)
    } yield State(
      s.params.copy(factorParams = params), s.factors, combineStates(state.map(_.toVector)))
  }

  /**
    * Remove rows from the factor loading matrix, beta, corresponding to 
    * partially missing observations
    */
  def missingBeta(
    y: DenseVector[Option[Double]],
    beta: DenseMatrix[Double]): DenseMatrix[Double] = {

    val nonMissing = KalmanFilter.indexNonMissing(y)
    beta(nonMissing.toVector, ::).toDenseMatrix
  }

  /**
    * Update the value of the variance in the
    * observation model, a diagonal matrix filled with identical
    * values
    * @param prior an inverse gamma
    * @param ys the observations
    * @param s the current state of the MCMC
    * @return the sampled value of sigma
    * TODO: Check this
    */
  def sampleSigma(
    prior:  InverseGamma,
    ys:     Vector[Dlm.Data],
    params: Parameters,
    fs:     Vector[(Double, Option[DenseVector[Double]])]): Rand[Double] = {

    val p = params.beta.rows
    val n = ys.size
    val shape = prior.shape + n * 0.5
    val fls = flattenFactors(fs).sortBy(_._1)
    val ssy = (ys zip fls).
      map {
        case (d, (timef, f)) =>
          val betam = missingBeta(d.observation, params.beta)
          val ym = KalmanFilter.flattenObs(d.observation)
          (ym - betam * f)
        case _ => DenseVector.zeros[Double](p)
      }.
      map(x => x *:* x).
      reduce(_ + _)

    val scale = prior.scale + 0.5 * mean(ssy)
    
    InverseGamma(shape, scale)
  }


  /**
    * Gibbs step for the factor stochastic volatility model
    */
  def sampleStep(
    priorBeta:     Gaussian,
    priorSigmaEta: InverseGamma,
    priorMu:       Gaussian,
    priorPhi:      Beta,
    priorSigma:    InverseGamma,
    observations:  Vector[Dlm.Data],
    p:             Int,
    k:             Int) = { (s: State) => 

    for {
      svp <- sampleVolatilityParams(priorMu, priorSigmaEta, priorPhi, p)(s)
      fs <- sampleFactors(observations, svp.params, svp.volatility)
      sigma <- sampleSigma(priorSigma, observations, s.params, fs)
      beta <- sampleBeta(priorBeta, observations, p, k, fs, s.params)
    } yield svp.copy(params = svp.params.copy(beta = beta, v = sigma), factors = fs)
  }

  /**
    * Initialise the factors
    * @param beta the factor loading matrix
    * @param observations a vector containing observations of the process
    * @param sigmaY The observation error variance
    * @return the factors sampled from a multivariate normal distribution
    */
  def initialiseFactors(
    beta:         DenseMatrix[Double],
    observations: Vector[Dlm.Data],
    sigmaY:       Double) = {

    val k = beta.cols
    val precY = diag(DenseVector.fill(beta.rows)(sigmaY))
    val obs = encodePartiallyMissing(observations)

    // sample factors independently
    val res = for {
      (time, yt) <- obs
      prec = DenseMatrix.eye[Double](k) + (beta.t * precY * beta)
      mean = yt map (y => prec \ (beta.t * (precY * y)))
      sample = mean map { m => rnorm(m, prec).draw  }
    } yield (time, sample)

    Rand.always(res)
  }

  /**
    * Initialise the state for the Factor SV model with AR(1) latent state
    */
  def initialiseStateAr(
    initP:        FactorSv.Parameters,
    observations: Vector[Dlm.Data],
    k:            Int) = {
    // initialise factors
    val factors = initialiseFactors(initP.beta, observations, initP.v).draw

    // initialise the latent state
    val initState = for {
      i <- Vector.range(0, k)
      fps = initP.factorParams(i)
      fs = extractFactors(factors, i)
      (mod, p) = StochasticVolatility.ar1Dlm(fps)
      state = StochasticVolatility.initialState(fps, fs, FilterAr.advanceState(fps),
        Smoothing.step(mod, p.w)).draw
    } yield state

    State(initP, factors, combineStates(initState.map(_.toVector)))
  }

  /**
    * Gibbs sampling for the factor stochastic volatility model with AR(1) latent-state
    * @param priorBeta the prior for the non-zero elements of the factor loading matrix
    * @param priorSigmaEta the prior for the state evolution noise
    * @param priorPhi the prior for the mean reverting factor phi
    * @param priorSigma the prior for the variance of the measurement noise
    * @param observations a vector of time series results
    * @param initP the parameters of the Factor model
    * @return a markov chain
    */
  def sampleAr(    
    priorBeta:     Gaussian,
    priorSigmaEta: InverseGamma,
    priorMu:       Gaussian,
    priorPhi:      Beta,
    priorSigma:    InverseGamma,
    observations:  Vector[Dlm.Data],
    initP:         FactorSv.Parameters): Process[State] = {

    // specify number of factors and time series
    val k = initP.beta.cols
    val p = initP.beta.rows

    // initialise the latent state 
    val init = initialiseStateAr(initP, observations, k)

    MarkovChain(init)(sampleStep(priorBeta, priorSigmaEta, priorMu, priorPhi,
      priorSigma, observations, p, k))
  }
}
