package core.dlm.model

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.distributions._
import breeze.numerics.{log, exp}
import cats.implicits._

/**
  * Simulate and fit a Stochastic volatility model using a mixture model approximation for
  * the non-linear dynamics
  * Y_t = sigma * exp(a_t / 2), sigma ~ N(0, 1)
  * a_t = phi * a_t + eta, eta ~ N(0, sigma_eta)
  */
object StochasticVolatility {
  private val pis = Array(0.0073, 0.1056, 0.00002, 0.044, 0.34, 0.2457, 0.2575)
  private val means = Array(-11.4, -5.24, -9.84, 1.51, -0.65, 0.53, -2.36)
  private val variances = Array(5.8, 2.61, 5.18, 0.17, 0.64, 0.34, 1.26)

  case class Parameters(
    phi:   Double,
    mu:    Double,
    sigmaEta: Double)

  /**
    * The observation function for the stochastic volatility model
    */
  def observation(at: Double): Rand[Double] = 
    Gaussian(0.0, 1).map(s => s * exp(at * 0.5))

  def stepState(p: Parameters, at: Double, dt: Double): ContinuousDistr[Double] = 
    Gaussian(p.mu + p.phi * (at - p.mu), p.sigmaEta * math.sqrt(dt))

  def simStep(
    time:  Double,
    p:     Parameters)(state: Double) = {
    for {
      x <- stepState(p, state, 1.0)
      y <- observation(x)
    } yield (time, y.some, x)
  }

  def simulate(p: Parameters) = {
    val initVar = p.sigmaEta * p.sigmaEta / (1 - p.phi * p.phi)
    val initState = Gaussian(p.mu, math.sqrt(initVar))
    val init = (0.0, None: Option[Double], initState.draw)

    MarkovChain(init)(a => simStep(a._1 + 1.0, p)(a._3))
  }

  /**
    * Sample the indices for the mixture model
    * @param ys a collection of observations
    * @param alphas the latent log-volatility
    */
  def sampleKt(
    ys:     Vector[(Double, Option[Double])],
    alphas: Vector[(Double, Double)]) = {

    // marginal likelihood
    def ll(j: Int, yo: Option[Double], x: Double) = {
      yo.map{ y =>
        Gaussian(x, math.sqrt(variances(j))).
          logPdf(log(y * y) - means(j))
      }.getOrElse(0.0)
    }

    // calculate the log weights for a single index
    def logWeights(yo: Option[Double], x: Double) = for {
      j <- 0 until variances.size
    } yield log(pis(j)) + ll(j, yo, x)

    for {
      (y, x) <- ys.map(_._2) zip alphas.tail.map(_._2)
      lw = logWeights(y, x)
      max = lw.max
      weights = lw.map(w => exp(w - max))
      kt = Multinomial(DenseVector(weights.toArray)).draw
    } yield kt
  }

  def ar1Dlm(params: Parameters): (Dlm.Model, Dlm.Parameters) = {
    // the AR(1) model as a DLM
    val mod = Dlm.Model(
      f = (t: Double) => DenseMatrix(1.0),
      g = (dt: Double) => DenseMatrix(params.phi))

    val c0 = params.sigmaEta * params.sigmaEta / (1 - params.phi * params.phi)

    val paramsSv = Dlm.Parameters(
      v = DenseMatrix(1.0),
      w = DenseMatrix(params.sigmaEta * params.sigmaEta),
      m0 = DenseVector.zeros[Double](1),
      c0 = DenseMatrix(c0)
    )

    (mod, paramsSv)
  }

  /**
    * Sample the log-variance using a mixture model approximation
    */
  def sampleState(
    ys:     Vector[(Double, Option[Double])],
    alphas: Vector[(Double, Double)],
    params: Parameters) = {

    val t0 = ys.head._1
    val dt0 = ys(1)._1 - ys.head._1
    val (mod, paramsSv) = ar1Dlm(params)

    // sample the T indices of the mixture
    val kt = sampleKt(ys, alphas)

    // construct a list of variances and means
    val vkt = kt.map(j => variances(j))
    val mkt = kt.map(j => means(j))

    val init = KalmanFilter.State(t0 - dt0, paramsSv.m0, paramsSv.c0,
      paramsSv.m0, paramsSv.c0, None, None)

    val yt = (ys zip mkt).
      map { case ((t, yo), m) => (t, yo map (y => log(y * y) - m)) }.
      map { case (t, x) => Dlm.Data(t, DenseVector(x)) }

    // create vector of parameters
    val ps = vkt map (newV => paramsSv.copy(v = DenseMatrix(newV)))

    val filtered = (ps zip yt).
      scanLeft(init){ case (s, (p, y)) => KalmanFilter.step(mod, p)(s, y) }

    Rand.always(Smoothing.sample(mod, filtered.toVector, paramsSv.w).
      map { case (t, x) => (t, x(0)) })
  }

  /**
    * Log-Likelihood of the AR(1) process
    * @param state the current value of the state
    * @param p the current value of the parameters
    * @return the log-likelihood
    */
  def arLikelihood(
    state: Vector[(Double, Double)],
    p: Parameters): Double = {
    val n = state.length

    val ssa = (state zip state.tail).
      map { case (at, at1) => at1._2 - (p.mu + p.phi * (at._2 - p.mu)) }.
      map (x =>  x * x).
      sum

    - n * 0.5 * log(2 * math.Pi * p.sigmaEta * p.sigmaEta) - (1/(2 * p.sigmaEta * p.sigmaEta)) * ssa
  }


  /**
    * Sample Phi using a Beta prior and proposal distribution
    * @param tau a small tuning parameter for the beta proposal
    * @param lambda a tuning parameter for the beta proposal distribution
    * @param prior a prior distribution for the parameter phi
    * @return a Metropolis Hastings step sampling the value of Phi
    */
  def samplePhi(
    tau:    Double,
    lambda: Double,
    prior:  ContinuousDistr[Double],
    p:      Parameters,
    alpha:  Vector[(Double, Double)]) = {

    val proposal = (phi: Double) => {
      new Beta(lambda * phi + tau, lambda * (1 - phi) + tau)
    }

    val pos = (phi: Double) => {
      prior.logPdf(phi) + arLikelihood(alpha, p.copy(phi = phi))
    }

    MarkovChain.Kernels.metropolis(proposal)(pos)
  }

  /**
    * Sample sigma from the an inverse gamma distribution (sqrt)
    * @param prior the prior for the variance of the noise of the latent-state
    * @return a distribution over the system variance
    */
  def sampleSigma(
    prior:  InverseGamma,
    p:      Parameters,
    alphas: Vector[(Double, Double)]) = { 

    val as = alphas.map { case (t, x) => (t, DenseVector(x)) }
    GibbsSampling.sampleSystemMatrix(prior, as, (t: Double) => DenseMatrix(p.phi)).
      map(d => math.sqrt(d(0, 0)))
  }

  /**
    * Transform the observations y to log(y^2) and remove the mean
    * @param ys a vector of univariate observations
    * @return a transformed vector of observations suitable for Kalman Filtering
    */
  def transformObservations(ys: Vector[(Double, Option[Double])]) = {
    for {
      (time, yo) <- ys
      yt = yo map (y => log(y * y) + 1.27)
    } yield Dlm.Data(time, DenseVector(yt))
  }


  /**
    * Sample the initial state from the Gaussian approximation of the SV model
    * log(y_t^2) = a_t + log(e_t^2)
    * a_t = phi a_t-1 + eta_t
    * specifying the latent state evolution
    * @param p the parameters of the stochastic volatility model
    * @param ys the time series observations: y_t = e_t * exp(a_t / 2)
    * @return the latent-state a_t under the assumption that log(e_t^2) is Gaussian
    */
  def initialState(
    params: Parameters,
    ys:     Vector[(Double, Option[Double])]) = {
   
    val yt = transformObservations(ys)
    val (mod, paramsSv) = ar1Dlm(params)

    Smoothing.ffbs(mod, yt, paramsSv.copy(v = DenseMatrix(math.Pi * math.Pi * 0.5))).
      map(_.map { case (t, x) => (t, x(0)) })
  }

  case class State(
    params: Parameters,
    alphas: Vector[(Double, Double)])

  def step(
    priorSigma: InverseGamma,
    priorPhi:   ContinuousDistr[Double],
    ys:         Vector[(Double, Option[Double])])(s: State) = {
    for {
      alphas <- sampleState(ys, s.alphas, s.params)
      newPhi <- samplePhi(0.05, 100, priorPhi, s.params, alphas)(s.params.phi)
      newSigma <- sampleSigma(priorSigma, s.params, alphas)
      p = Parameters(newPhi, 0.0, newSigma)
    } yield State(p, alphas)
  }

  def sample(
    priorSigma: InverseGamma,
    priorPhi:   ContinuousDistr[Double],
    params:     Parameters,
    ys:         Vector[(Double, Option[Double])]) = {

    // initialise the latent state
    val alphas = initialState(params, ys).draw
    val init = State(params, alphas)

    MarkovChain(init)(step(priorSigma, priorPhi, ys))
  }
}
