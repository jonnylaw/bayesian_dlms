package core.dlm.model

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.distributions._
import breeze.numerics.{log, exp}
import cats.implicits._

case class SvParameters(phi: Double, mu: Double, sigmaEta: Double) {
  def toList: List[Double] = phi :: mu :: sigmaEta :: Nil
}

/**
  * Simulate and fit a Stochastic volatility model using a mixture model approximation for
  * the non-linear dynamics for either a AR(1) latent-state or OU latent state
  * 
  * Y_t = sigma * exp(a_t / 2), sigma ~ N(0, 1)
  * a_t = phi * a_t + eta, eta ~ N(0, sigma_eta)
  */
object StochasticVolatility extends {
  private val pis = Array(0.0073, 0.1056, 0.00002, 0.044, 0.34, 0.2457, 0.2575)
  private val means = Array(-11.4, -5.24, -9.84, 1.51, -0.65, 0.53, -2.36)
  private val variances = Array(5.8, 2.61, 5.18, 0.17, 0.64, 0.34, 1.26)

  case class State(params: SvParameters, alphas: Vector[(Double, Double)])

  /**
    * The observation function for the stochastic volatility model
    */
  def observation(at: Double): Rand[Double] =
    Gaussian(0.0, 1).map(s => s * exp(at * 0.5))

  def stepState(p: SvParameters,
                at: Double,
                dt: Double): ContinuousDistr[Double] =
    Gaussian(p.mu + p.phi * (at - p.mu), p.sigmaEta * math.sqrt(dt))

  def simStep(time: Double, p: SvParameters)(state: Double) = {
    for {
      x <- stepState(p, state, 1.0)
      y <- observation(x)
    } yield (time, y.some, x)
  }

  def simulate(p: SvParameters) = {
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
  def sampleKt(ys: Vector[(Double, Option[Double])],
               alphas: Vector[(Double, Double)]) = {

    // marginal likelihood
    def ll(j: Int, yo: Option[Double], x: Double) = {
      yo.map { y =>
          Gaussian(x, math.sqrt(variances(j))).logPdf(log(y * y) - means(j))
        }
        .getOrElse(0.0)
    }

    // calculate the log weights for a single index
    def logWeights(yo: Option[Double], x: Double) =
      for {
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

  def ar1Dlm(params: SvParameters): (DlmModel, DlmParameters) = {
    // the AR(1) model as a DLM
    val mod = DlmModel(f = (t: Double) => DenseMatrix(1.0),
                       g = (dt: Double) => DenseMatrix(params.phi))

    val c0 = params.sigmaEta * params.sigmaEta / (1 - params.phi * params.phi)

    val paramsSv = DlmParameters(
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
    ys: Vector[(Double, Option[Double])],
    alphas: Vector[(Double, Double)],
    params: SvParameters,
    advState: (KfState, Double) => KfState,
    backStep: (KfState, SamplingState) => SamplingState) = {

    val t0 = ys.head._1
    val dt0 = ys(1)._1 - ys.head._1
    val (mod, paramsSv) = ar1Dlm(params)

    // sample the T indices of the mixture
    val kt = sampleKt(ys, alphas)

    // construct a list of variances and means
    val vkt = kt.map(j => variances(j))
    val mkt = kt.map(j => means(j))

    val init = KfState(t0 - dt0,
                       paramsSv.m0,
                       paramsSv.c0,
                       paramsSv.m0,
                       paramsSv.c0,
                       None,
                       None,
                       0.0)

    val yt = (ys zip mkt)
      .map { case ((t, yo), m) => (t, yo map (y => log(y * y) - m)) }
      .map { case (t, x) => Dlm.Data(t, DenseVector(x)) }

    // create vector of parameters
    val ps = vkt map (newV => paramsSv.copy(v = DenseMatrix(newV)))

    val filtered = (ps zip yt).scanLeft(init) {
      case (s, (p, y)) => KalmanFilter.step(mod, p, advState)(s, y)
    }

    Rand.always(Smoothing.sample(mod, filtered.toVector, paramsSv.w, backStep).map {
      case (t, x) => (t, x(0))
    })
  }

  /**
    * Log-Likelihood of the AR(1) process
    * @param state the current value of the state
    * @param p the current value of the parameters
    * @return the log-likelihood
    */
  def arLikelihood(state: Vector[(Double, Double)], p: SvParameters): Double = {
    val n = state.length

    val ssa = (state zip state.tail)
      .map { case (at, at1) => at1._2 - (p.mu + p.phi * (at._2 - p.mu)) }
      .map(x => x * x)
      .sum

    -n * 0.5 * log(2 * math.Pi * p.sigmaEta * p.sigmaEta) - (1 / (2 * p.sigmaEta * p.sigmaEta)) * ssa
  }

  /**
    * Sample Phi using a Beta prior and proposal distribution
    * @param tau a small tuning parameter for the beta proposal
    * @param lambda a tuning parameter for the beta proposal distribution
    * @param prior a prior distribution for the parameter phi
    * @return a Metropolis Hastings step sampling the value of Phi
    */
  def samplePhi(tau: Double,
                lambda: Double,
                prior: ContinuousDistr[Double],
                p: SvParameters,
                alpha: Vector[(Double, Double)]) = {

    val proposal = (phi: Double) => {
      new Beta(lambda * phi + tau, lambda * (1 - phi) + tau)
    }

    val pos = (phi: Double) => {
      prior.logPdf(phi) + arLikelihood(alpha, p.copy(phi = phi))
    }

    MarkovChain.Kernels.metropolis(proposal)(pos)
  }

  /**
    * Sample mu from the autoregressive state space,
    * from a Gaussian distribution 
    * @param prior a Gaussian prior for the parameter
    * @return a function from the current state of the Markov chain to
    * a new state with a new mu sampled from the Gaussian posterior distribution
    */
  def sampleMu(
    prior:  Gaussian,
    p:      SvParameters,
    alphas: Vector[(Double, Double)]): Rand[Double] = {

    val n = alphas.tail.size
    val pmu = prior.mean
    val psigma = prior.variance

    val sumStates = (alphas.init, alphas.tail).zipped.
      map { case (at1, at) => (at._2 - p.phi * at1._2) }.reduce(_ + _)

    val prec = 1 / psigma + (n - 1) * (1 - p.phi) * (1 - p.phi) / p.sigmaEta * p.sigmaEta
    val mean = (pmu / psigma + ((1 - p.phi) / p.sigmaEta * p.sigmaEta) * sumStates) / prec
    val variance = 1 / prec

    Gaussian(mean, math.sqrt(variance))
  }

  /**
    * Sample sigma from the an inverse gamma distribution (sqrt)
    * @param prior the prior for the variance of the noise of the latent-state
    * @return a distribution over the system variance
    */
  def sampleSigma(prior: InverseGamma,
                  p: SvParameters,
                  alphas: Vector[(Double, Double)]) = {

    val as = alphas.map { case (t, x) => (t, DenseVector(x)) }

    GibbsSampling
      .sampleSystemMatrix(prior, as, (t: Double) => DenseMatrix(p.phi))
      .map(d => math.sqrt(d(0, 0)))
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
    params: SvParameters,
    ys: Vector[(Double, Option[Double])],
    advState: (KfState, Double) => KfState,
    backStep: (KfState, SamplingState) => SamplingState) = {

    val yt = transformObservations(ys)
    val (mod, paramsSv) = ar1Dlm(params)

    Smoothing
      .ffbs(mod, yt, advState, backStep, paramsSv.copy(v = DenseMatrix(math.Pi * math.Pi * 0.5)))
      .map(_.map { case (t, x) => (t, x(0)) })
  }

  /**
    * 
    */
  def stepAr(
    priorSigma: InverseGamma,
    priorPhi:   ContinuousDistr[Double],
    priorMu:    Gaussian,
    ys:         Vector[(Double, Option[Double])])(s: State) = {

    for {
      alphas <- sampleState(ys, s.alphas, s.params,
        FilterAr.advanceState(s.params), FilterAr.backwardStep(s.params))
      newPhi <- samplePhi(0.05, 100, priorPhi, s.params, alphas)(s.params.phi)
      newSigma <- sampleSigma(priorSigma, s.params.copy(phi = newPhi), alphas)
      newMu <- sampleMu(priorMu, s.params.copy(phi = newPhi, sigmaEta = newSigma), alphas)
      p = SvParameters(newPhi, newMu, newSigma)
    } yield State(p, alphas)
  }

  def sampleAr(
    priorSigma: InverseGamma,
    priorPhi:   ContinuousDistr[Double],
    priorMu:    Gaussian,
    params:     SvParameters,
    ys:         Vector[(Double, Option[Double])]) = {

    // initialise the latent state
    val alphas = initialState(params, ys,
      FilterAr.advanceState(params), FilterAr.backwardStep(params)).draw
    val init = State(params, alphas)

    MarkovChain(init)(stepAr(priorSigma, priorPhi, priorMu, ys))
  }

  /**
    * Marginal log likelihood of the OU process used to
    * perform the Metropolis Hastings steps to learn the static parameters
    * @param s the current state of the MCMC including the current values
    * of the parameters
    * @return the log likelihood of the state given the static parameter values
    */
  def ouLikelihood(
    p: SvParameters,
    alphas: Vector[(Double, Double)]): Double = {

    val phi = (dt: Double) => exp(-p.phi * dt)
    val variance = (dt: Double) =>
      (math.pow(p.sigmaEta, 2) / (2*p.phi)) * (1 - exp(-2*p.phi*dt))

    (alphas.init zip alphas.tail).map { case (a0, a1) =>
      val dt = a1._1 - a0._1
      if (dt == 0.0) {
        0.0
      } else {
        val mean = p.mu + phi(dt) * (a0._2 - p.mu)
        Gaussian(mean, math.sqrt(variance(dt))).logPdf(a1._2)
      }
    }.sum
  }

  /**
    * Sample the mean from the OU process using a metropolis step
    * @param prior a prior distribution for the mean parameter, mu
    * @param delta the standard deviation of the (Gaussian) proposal distribution
    * @return a function from the current state of the MCMC algorithm to
    * Rand[GibbsUnivariate.State] including updating of the accepted field
    */
  def sampleMuOu(
    prior:  ContinuousDistr[Double],
    delta:  Double = 0.05,
    p:      SvParameters,
    alphas: Vector[(Double, Double)]) = {

    val proposal = (mu: Double) => Gaussian(mu, delta)

    val pos = (newMu: Double) => {
      val newP = p.copy(mu = newMu)
      prior.logPdf(newMu) + ouLikelihood(newP, alphas)
    }

    MarkovChain.Kernels.metropolis(proposal)(pos)
  }

  /**
    * Metropolis step to sample the value of sigma_eta
    */
  def sampleSigmaMetropOu(
    prior: ContinuousDistr[Double],
    delta: Double = 0.05,
    p: SvParameters,
    alphas: Vector[(Double, Double)]) = {

    val proposal = (sigmaEta: Double) => for {
      z <- Gaussian(0.0, delta)
      newSigma = sigmaEta * exp(z)
    } yield newSigma

    val pos = (newSigma: Double) => {
      val newP = p.copy(sigmaEta = newSigma)
      prior.logPdf(newSigma) + ouLikelihood(newP, alphas)
    }

    MarkovChain.Kernels.metropolis(proposal)(pos)
  }

  /**
    * Gibbs step for updating sigma in the OU process
    */
  def sampleSigmaOu(
    prior: InverseGamma,
    alphas: Vector[(Double, Double)],
    p: SvParameters) = {

    // take the squared difference of x_t - g(x_{t-1}) for t = 1 ... T
    // of the sampled state
    // add them all up
    val squaredSum = (alphas.tail zip alphas.init).
      map { case (mt, at) =>
        val dt = mt._1 - at._1
        if (dt == 0.0) {
          0.0
        } else {
          val at1 = p.mu + exp(-p.phi * dt) * (at._2 - p.mu)
          ((mt._2 - at1) * (mt._2 - at1) * 2 * p.phi) / (1 - exp(-2 * p.phi * dt))
        }
      }.
      sum

    val shape = prior.shape + (alphas.size - 1) * 0.5
    val rate = prior.scale + squaredSum * 0.5

    InverseGamma(shape, rate)
  }

  /**
    * Sample the autoregressive parameter (between 0 and 1)
    */
  def samplePhiOu(
    prior:  ContinuousDistr[Double],
    p:      SvParameters,
    alphas: Vector[(Double, Double)],
    lambda: Double = 10.0,
    tau:    Double = 0.05) = {

    val proposal = (phi: Double) =>
      new Beta(lambda * phi + tau, lambda * (1 - phi) + tau)

    val pos = (newPhi: Double) => {
      val newP = p.copy(phi = newPhi)
      prior.logPdf(newPhi) + ouLikelihood(newP, alphas)
    }

    MarkovChain.Kernels.metropolis(proposal)(pos)
  }

  def stepOu(
    priorSigma: InverseGamma,
    priorPhi:   ContinuousDistr[Double],
    priorMu:    Gaussian,
    ys:         Vector[(Double, Option[Double])])(s: State) = {

    for {
      alphas <- sampleState(ys, s.alphas, s.params,
        FilterOu.advanceState(s.params), FilterOu.backwardStep(s.params))
      newPhi <- samplePhiOu(priorPhi, s.params, alphas, 0.05, 100)(s.params.phi)
      newSigma <- sampleSigmaOu(priorSigma, alphas, s.params.copy(phi = newPhi))
      newMu <- sampleMuOu(priorMu, 0.05, s.params.copy(phi = newPhi, sigmaEta = newSigma), alphas)(s.params.mu)
      p = SvParameters(newPhi, newMu, newSigma)
    } yield State(p, alphas)

  }

  def sampleOu(
    priorSigma: InverseGamma,
    priorPhi:   ContinuousDistr[Double],
    priorMu:    Gaussian,
    params:     SvParameters,
    ys:         Vector[(Double, Option[Double])]) = {

    // initialise the latent state
    val alphas = initialState(params, ys,
      FilterOu.advanceState(params), FilterOu.backwardStep(params)).draw
    val init = State(params, alphas)

    MarkovChain(init)(stepOu(priorSigma, priorPhi, priorMu, ys))
  }
}
