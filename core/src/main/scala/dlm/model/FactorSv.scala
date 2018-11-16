package dlm.core.model

import breeze.stats.distributions._
import breeze.stats.distributions._
import breeze.numerics.exp
import breeze.linalg.{DenseVector, DenseMatrix, diag, svd}
import breeze.linalg.svd._
import cats.implicits._

/**
  * Factor Stochastic Volatility Parameters for a model with
  * k factors and p time series
  * k << p
  * @param v the variance of the measurement error
  * @param beta the factor loading matrix, p x k
  * @param factorParams a vector of length k containing the
  * ISV parameters for the factors
  */
case class FsvParameters(
  v:            DenseMatrix[Double],
  beta:         DenseMatrix[Double],
  factorParams: Vector[SvParameters]) {

  def diagonal(m: DenseMatrix[Double]): List[Double] = {
    for {
      i <- List.range(0, m.rows)
    } yield m(i, i)
  }

  def toList = diagonal(v) ::: beta.data.toList ::: factorParams.flatMap(_.toList).toList

  def map(f: Double => Double) =
    FsvParameters(v map f, beta.map(f), factorParams.map(_.map(f)))

  def add(p: FsvParameters): FsvParameters =
    FsvParameters(v + p.v, p.beta + beta,
      (factorParams zip p.factorParams).
        map { case (f, f1) => f add f1 })
}

object FsvParameters {
  def apply(v: Double, beta: DenseMatrix[Double],
            factorParams: Vector[SvParameters]): FsvParameters = {
    FsvParameters(diag(DenseVector.fill(beta.rows)(v)), beta, factorParams)
  }

  def empty(p: Int, k: Int): FsvParameters =
    FsvParameters(DenseMatrix.eye[Double](p), FactorSv.makeBeta(p, k),
      Vector.fill(k)(SvParameters.empty))

  def fromList(p: Int, k: Int)(l: List[Double]): FsvParameters =
    FsvParameters(diag(DenseVector(l.take(p).toArray)),
      new DenseMatrix(p, k, l.slice(1, p + p * k).toArray),
      l.drop(p + p * k).
        grouped(3).
        map(SvParameters.fromList).
        toVector
    )
}

/**
  * Model a large covariance matrix using a factor structure
  */
object FactorSv {
  /**
    * The state of the Gibbs Sampler
    * @param p the current sample of the fsv parameters
    * @param factors the current sample of the time series of factors, k x n dimensional
    * @param volatility the current sample of the time series of variances, k x n dimensional
    */
  case class State(
    params:     FsvParameters,
    factors:    Vector[(Double, Option[DenseVector[Double]])],
    volatility: Vector[SamplingState]
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
    params: FsvParameters)(a: Vector[Double]) = {

    for {
      vt <- MultivariateGaussian(
        DenseVector.zeros[Double](params.beta.rows),
        params.v)

      fs <- (a zip params.factorParams).
        traverse { case (x, p) => StochasticVolatility.simStep(t, p)(x) }

      f = fs.map { case (t, ft, at) => ft.get }

      a1 = fs.map { case (t, ft, at) => at }

      y = params.beta * DenseVector(f.toArray) + vt
    } yield (Data(t, y.map(_.some)), f, a1)
  }

  def simulate(p: FsvParameters) = {
    val k = p.beta.cols
    val initState = Vector.fill(k)(Gaussian(0.0, 1.0).draw)
    val init = (Data(0.0, DenseVector[Option[Double]](None)),
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
    obs: Vector[Data]): Vector[(Double, Option[DenseVector[Double]])] = {

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
    observations: Vector[Data],
    p: FsvParameters,
    volatility: Vector[SamplingState]) = {

    val precY = diag(diag(p.v).map(1.0 / _))
    val beta = p.beta
    val obs = encodePartiallyMissing(observations)

    // sample factors independently
    val res = for {
      ((time, ys), at) <- obs zip volatility.tail
      fVar = diag(exp(-at.sample))
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
  def getithObs(y: Vector[Data], i: Int): Vector[Option[Double]] = {
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
    observations: Vector[Data],
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
    prior: Gaussian,
    ys:    Vector[Data],
    p:     Int,
    k:     Int,
    fs:    Vector[(Double, Option[DenseVector[Double]])],
    v:     DenseMatrix[Double]): DenseMatrix[Double] = {

    val newbeta = makeBeta(p, k)
    val fls = flattenFactors(fs.sortBy(_._1)).map(_._2)

    // mutate the beta matrix
    (1 until p).foreach { i =>
      if (i < k) {
        newbeta(i, 0 until i).t := sampleBetaRow(prior, fls, ys, v(i, i), i, k)
      } else {
        newbeta(i, ::).t := sampleBetaRow(prior, fls, ys, v(i, i), i, k)
      }}

    newbeta
  }

  /**
    * Extract a single state from a vector of states
    * @param vs the combined state
    * @param i the position of the state to extract
    * @return the extracted state
    */
  def extractState(
    vs: Vector[SamplingState],
    i: Int): Vector[FilterAr.SampleState] =
    vs.map { s => FilterAr.SampleState(s.time, s.sample(i),
      s.mean(i), s.cov(i,i), s.at1(i), s.rt1(i,i))
    }

  def combineStates(s: Vector[Vector[FilterAr.SampleState]]): Vector[SamplingState] =
    s.transpose.
      map { x =>
        SamplingState(
          time = x.head.time,
          sample = DenseVector(x.map(_.sample).toArray),
          mean = DenseVector(x.map(_.mean).toArray),
          cov = diag(DenseVector(x.map(_.cov).toArray)),
          at1 = DenseVector(x.map(_.at1).toArray),
          rt1 = diag(DenseVector(x.map(_.rt1).toArray)))
      }
  
  /**
    * Extract the ith factor from a multivariate vector of factors
    */
  def extractFactors(
    fs: Vector[(Double, Option[DenseVector[Double]])],
    i:  Int): Vector[(Double, Option[Double])] = {

    fs map { case (t, fo) => (t, fo.map(f => f(i))) }
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
  def sampleVolatilityParamsAr(
    priorPhi:      Gaussian,
    priorMu:       Gaussian,
    priorSigmaEta: InverseGamma,
    p:             Int)(s: State) = {

    val k = s.params.beta.cols

    val res = for {
      i <- Vector.range(0, k)
      thisState = extractState(s.volatility, i)
      theseParameters = s.params.factorParams(i)
      factorState = StochVolState(theseParameters, thisState)
      theseFactors = extractFactors(s.factors, i)
      factor = StochasticVolatility.stepUni(priorPhi,
        priorMu, priorSigmaEta, theseFactors)(factorState)
    } yield factor

    for {
      res <- res.sequence
      params = res.map(_.params)
      state = res.map(_.alphas)
    } yield State(
      s.params.copy(factorParams = params), s.factors, combineStates(state.map(_.toVector)))
  }

  /**
    * Sample the log-volatility of a factor model
    */
  def sampleVolatilityAr(
    p: Int,
    params: FsvParameters,
    factors: Vector[(Double, Option[DenseVector[Double]])],
    volatility: Vector[SamplingState]
  ): Rand[Vector[SamplingState]] = {

    val k = params.beta.cols

    val res = for {
      i <- Vector.range(0, k)
      thisState = extractState(volatility, i)
      theseParameters = params.factorParams(i)
      theseFactors = extractFactors(factors, i)
      logVol = StochasticVolatility.sampleStateAr(theseFactors,
        theseParameters, thisState)
    } yield logVol

    res.sequence.map(combineStates)
  }

  /**
    * Sample each of the factor states and parameters in turn
    * @param priorMu a Normal prior for the mean of the OU
    * @param priorSigmaEta an inverse Gamma prior for the variance of the OU process
    * @param priorPhi a Beta prior on the mean reversion parameter phi
    * @param p an integer specifying the dimension of the observations
    * @param s the current state of the MCMC algorithm
    * @return
    */
  def sampleVolatilityParamsOu(
    priorMu:       Gaussian,
    priorSigmaEta: InverseGamma,
    priorPhi:      Beta,
    p:             Int)(s: State) = {

    val k = s.params.beta.cols

    val res = for {
      i <- Vector.range(0, k).par
      thisState = extractState(s.volatility, i)
      theseParameters = s.params.factorParams(i)
      factorState = StochVolState(theseParameters, thisState)
      theseFactors = extractFactors(s.factors, i)
      factor = StochasticVolatility.stepOu(priorPhi, priorMu,
        priorSigmaEta, theseFactors)(factorState)
    } yield factor

    for {
      res <- res.seq.sequence
      params = res.map(_.params)
      state = res.map(_.alphas)
    } yield State(
      s.params.copy(factorParams = params), s.factors,
      combineStates(state.map(_.toVector)))
  }

  /**
    * Sample the log-volatility of a factor model
    */
  def sampleVolatilityOu(
    p: Int,
    params: FsvParameters,
    factors: Vector[(Double, Option[DenseVector[Double]])],
    volatility: Vector[SamplingState]
  ): Vector[SamplingState] = {

    val k = params.beta.cols

    val res: Vector[Vector[FilterAr.SampleState]] = for {
      i <- Vector.range(0, k)
      thisState = extractState(volatility, i)
      theseParameters = params.factorParams(i)
      theseFactors = extractFactors(factors, i)
      knots = StochasticVolatilityKnots.sampleKnots(10, 100, theseFactors.size).draw
      logVol = StochasticVolatilityKnots.sampleState(
        StochasticVolatilityKnots.ffbsOu,
        StochasticVolatilityKnots.filterOu,
        StochasticVolatilityKnots.sampleOu)(theseFactors, theseParameters,
          knots, thisState.toArray)
    } yield logVol.toVector

    combineStates(res)
  }

  /**
    * Replace rows from the factor loading matrix with zeros, beta,
    * corresponding to 
    * partially missing observations
    */
  def missingBeta(
    y: DenseVector[Option[Double]],
    beta: DenseMatrix[Double]): DenseMatrix[Double] = {

    val nonMissing = KalmanFilter.indexNonMissing(y)
    nonMissing.toVector foreach (i =>
      beta(i, ::).t := DenseVector.zeros[Double](beta.cols))

    beta
  }

  /**
    * Update the value of the variance in the
    * observation model, a diagonal matrix filled with identical
    * values
    * @param prior an inverse gamma
    * @param ys the observations
    * @param s the current state of the MCMC
    * @return the sampled value of sigma
    */
  def sampleSigmaUni(
    prior:  InverseGamma,
    ys:     Vector[Data],
    params: FsvParameters,
    fs:     Vector[(Double, Option[DenseVector[Double]])]): Rand[Double] = {

    val n = ys.size
    val shape = prior.shape + n * 0.5

    val fls = flattenFactors(fs).sortBy(_._1)
    val ssy = (ys zip fls).
      map {
        case (d, (timef, f)) =>
          val betam = missingBeta(d.observation, params.beta)
          val ym = KalmanFilter.flattenObs(d.observation)
          val centered = (ym - betam * f)
          centered.t * centered
        case _ => 0.0
      }.
      reduce(_ + _)

    val scale = prior.scale + 0.5 * ssy

    InverseGamma(shape, scale)
  }

  /**
    * Update the value of the variance in the
    * observation model, a diagonal matrix filled with identical
    * values
    * @param prior an inverse gamma
    * @param ys the observations
    * @param s the current state of the MCMC
    * @return the sampled value of sigma
    */
  def sampleSigma(
    prior:  InverseGamma,
    ys:     Vector[Data],
    params: FsvParameters,
    fs:     Vector[(Double, Option[DenseVector[Double]])]): Rand[DenseMatrix[Double]] = {

    val ns: Vector[Int] = for {
      i <- Vector.range(0, ys.head.observation.size)
      n = ys.map(_.observation(i)).flatten.size
    } yield n

    val fls = flattenFactors(fs).sortBy(_._1)
    val ssy: DenseVector[Double] = (ys zip fls).
      map {
        case (d, (timef, f)) =>
          val betam = missingBeta(d.observation, params.beta)
          val fi = betam * f
          val res = d.observation.data.zipWithIndex.map {
            case (Some(y), i) =>
              val centered = y - fi(i)
              centered * centered
            case _ => 0.0
          }
          DenseVector(res)
      }.
      reduce(_ + _)

    val res: Vector[Rand[Double]] = for {
      (ss, n) <- ssy.data.toVector zip ns
      shape = prior.shape + n * 0.5
      scale = prior.scale + 0.5 * ss
    } yield InverseGamma(shape, scale)

    res.sequence.map(vs => diag(DenseVector(vs.toArray)))
  }

  /**
    * Gibbs step for the factor stochastic volatility model
    */
  def sampleStep(
    priorBeta:     Gaussian,
    priorSigmaEta: InverseGamma,
    priorMu:       Gaussian,
    priorPhi:      Gaussian,
    priorSigma:    InverseGamma,
    observations:  Vector[Data],
    p:             Int,
    k:             Int)(s: State): Rand[State] = {

    for {
      svp <- sampleVolatilityParamsAr(priorPhi, priorMu, priorSigmaEta, p)(s)
      fs <- sampleFactors(observations, svp.params, svp.volatility)
      sigma <- sampleSigma(priorSigma, observations, svp.params, fs)
      newBeta = sampleBeta(priorBeta, observations, p, k, fs, svp.params.v)
    } yield State(svp.params.copy(v = sigma, beta = newBeta),
                  fs, svp.volatility)
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
    observations: Vector[Data],
    sigmaY:       DenseMatrix[Double]) = {

    val k = beta.cols
    val precY = diag(diag(sigmaY).map(1.0 / _))
    val obs = encodePartiallyMissing(observations)

    // sample factors independently
    val res = for {
      (time, yt) <- obs
      prec = DenseMatrix.eye[Double](k) + (beta.t * precY * beta)
      mean = yt map (y => prec \ (beta.t * (precY * y)))
      sample = mean map { m => rnorm(m, prec).draw }
    } yield (time, sample)

    Rand.always(res)
  }

  /**
    * Initialise the state for the Factor SV model with AR(1) latent state
    */
  def initialiseStateAr(
    initP:        FsvParameters,
    observations: Vector[Data],
    k:            Int) = {
    // initialise factors
    val factors = initialiseFactors(initP.beta, observations, initP.v).draw

    // initialise the latent state
    val initState = for {
      i <- Vector.range(0, k)
      fps = initP.factorParams(i)
      fs = extractFactors(factors, i)
      state = StochasticVolatilityKnots.initialStateAr(fps, fs)
    } yield state

    State(initP, factors, combineStates(initState.map(_.draw)))
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
    priorPhi:      Gaussian,
    priorSigma:    InverseGamma,
    observations:  Vector[Data],
    initP:         FsvParameters): Process[State] = {

    // specify number of factors and time series
    val k = initP.beta.cols
    val p = initP.beta.rows

    // initialise the latent state
    val init = initialiseStateAr(initP, observations, k)

    MarkovChain(init)(sampleStep(priorBeta, priorSigmaEta, priorMu, priorPhi,
      priorSigma, observations, p, k))
  }
 
  /**
    * Initialise the state for the Factor SV model with OU latent state
    */
  def initialiseStateOu(
    initP:        FsvParameters,
    observations: Vector[Data],
    k:            Int) = {
    // initialise factors
    val factors = initialiseFactors(initP.beta, observations, initP.v).draw

    // initialise the latent state
    val initState = for {
      i <- Vector.range(0, k)
      fps = initP.factorParams(i)
      fs = extractFactors(factors, i)
      state = StochasticVolatility.initialStateOu(fps, fs)
    } yield state

    State(initP, factors, combineStates(initState.map(_.draw)))
  }

  def stepOu(
    priorBeta:     Gaussian,
    priorSigmaEta: InverseGamma,
    priorMu:       Gaussian,
    priorPhi:      Beta,
    priorSigma:    InverseGamma,
    observations:  Vector[Data],
    p:             Int,
    k:             Int) = { (s: State) =>

    for {
      svp <- sampleVolatilityParamsOu(priorMu, priorSigmaEta, priorPhi, p)(s)
      fs <- sampleFactors(observations, svp.params, svp.volatility)
      sigma <- sampleSigma(priorSigma, observations, svp.params, fs)
      beta = sampleBeta(priorBeta, observations, p, k, fs, sigma)
    } yield svp.copy(params = svp.params.copy(beta = beta, v = sigma), factors = fs)
  }

  def sampleOu(
    priorBeta:     Gaussian,
    priorSigmaEta: InverseGamma,
    priorMu:       Gaussian,
    priorPhi:      Beta,
    priorSigma:    InverseGamma,
    observations:  Vector[Data],
    initP:         FsvParameters): Process[State] = {

    // specify number of factors and time series
    val p = initP.beta.rows
    val k = initP.beta.cols

    // initialise the latent state 
    val init = initialiseStateAr(initP, observations, k)

    MarkovChain(init)(stepOu(priorBeta, priorSigmaEta, priorMu, priorPhi,
      priorSigma, observations, p, k))
  }
}
