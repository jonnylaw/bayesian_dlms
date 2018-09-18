package dlm.core.model

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
object StochasticVolatility {
  private val pis = Array(0.0073, 0.1056, 0.00002, 0.044, 0.34, 0.2457, 0.2575)
  private val means = Array(-11.4, -5.24, -9.84, 1.51, -0.65, 0.53, -2.36)
  private val variances = Array(5.8, 2.61, 5.18, 0.17, 0.64, 0.34, 1.26)

  /**
    * The observation function for the stochastic volatility model
    */
  def observation(at: Double): Rand[Double] =
    Gaussian(0.0, 1).map(s => s * exp(at * 0.5))

  def stepState(
    p:  SvParameters,
    at: Double,
    dt: Double): ContinuousDistr[Double] =
    Gaussian(p.mu + p.phi * (at - p.mu), p.sigmaEta * math.sqrt(dt))

  def simStep(time: Double, p: SvParameters)(state: Double) = {
    for {
      x <- stepState(p, state, 1.0)
      y <- observation(x)
    } yield (time, y.some, x)
  }

  /**
    * Simulate regularly from a stochastic volatility model with
    * AR(1) latent-state
    */
  def simulate(p: SvParameters) = {
    val initVar = p.sigmaEta * p.sigmaEta / (1 - p.phi * p.phi)
    val initState = Gaussian(p.mu, math.sqrt(initVar))
    val init = (0.0, None: Option[Double], initState.draw)

    MarkovChain(init)(a => simStep(a._1 + 1.0, p)(a._3))
  }

  /**
    * Advance the state of the OU process
    */
  def stepOu(
    p: SvParameters,
    at: Double,
    dt: Double): ContinuousDistr[Double] = {

    val mean = at * exp(-p.phi * dt) + p.mu * (1 - exp(-p.phi * dt))
    val variance = (math.pow(p.sigmaEta, 2) / (2*p.phi)) * (1 - exp(-2*p.phi*dt))

    Gaussian(mean, math.sqrt(variance))
  }

  /**
    * Simulate from a Stochastic Volatility model with Ornstein-Uhlenbeck latent State
    */
  def simOu(
    p:     SvParameters,
    times: Stream[Double]): Stream[(Double, Option[Double], Double)] = {

    val initVar = p.sigmaEta * p.sigmaEta / (1 - p.phi * p.phi)
    val initState = Gaussian(p.mu, math.sqrt(initVar))
    val init = (0.0, None: Option[Double], initState.draw)

    times.tail.scanLeft(init) { case (d0, t) =>
      val dt = t - d0._1
      val a1 = stepOu(p, d0._3, dt).draw
      val y = observation(a1).draw
      (t, y.some, a1)
    }
  }

  /**
    * Sample the indices for the mixture model
    * @param ys a collection of observations
    * @param alphas the latent log-volatility
    */
  def sampleKt(ys: Vector[Data],
               alphas: Vector[SamplingState]) = {

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
      (y, x) <- ys.map(_.observation) zip alphas.tail.map(_.sample(0))
      lw = logWeights(y(0), x)
      max = lw.max
      weights = lw.map(w => exp(w - max))
      kt = Multinomial(DenseVector(weights.toArray)).draw
    } yield kt
  }

  def ar1DlmParams(params: SvParameters): DlmParameters = {
    val c0 = params.sigmaEta * params.sigmaEta / (1 - params.phi * params.phi)

    DlmParameters(
      v = DenseMatrix(1.0),
      w = DenseMatrix(params.sigmaEta * params.sigmaEta),
      m0 = DenseVector(params.mu),
      c0 = DenseMatrix(c0)
    )
  }

  /**
    * Sample the log-variance using a mixture model approximation
    */
  def sampleState(
    ys:       Vector[Data],
    params:   DlmParameters,
    phi:      Double,
    advState: (KfState, Double) => KfState,
    backStep: (KfState, SamplingState) => SamplingState)(
    alphas:   Vector[SamplingState])= {

    val t0 = ys.head.time
    val dt0 = ys(1).time - ys.head.time
    val mod = Dlm.autoregressive(phi)

    // sample the T indices of the mixture
    val kt = sampleKt(ys, alphas)

    // construct a list of variances and means
    val vkt = kt.map(j => variances(j))
    val mkt = kt.map(j => means(j))

    val init = KfState(
      t0 - dt0,
      params.m0,
      params.c0,
      params.m0,
      params.c0,
      None,
      None,
      0.0)

    val yt = (ys zip mkt)
      .map { case (d, m) =>
        d.copy(observation = DenseVector(d.observation(0).map(y => log(y * y) - m))) }

    // create vector of parameters
    val ps = vkt map (newV => params.copy(v = DenseMatrix(newV)))

    val filtered = (ps zip yt).scanLeft(init) {
      case (s, (p, y)) => KalmanFilter(advState).step(mod, p)(s, y)
    }

    val initState = Smoothing.initialise(filtered)
    Rand.always(filtered.scanRight(initState)(backStep))
  }

  /**
    * Sample the log-volatility using a mixture model approximation and
    * the SVD FFBS algorithm
    * @param ys a vector of observations
    * @param alphas the current value of the 
    */
  def sampleStateSvd(
    ys: Vector[Data],
    alphas: Vector[SamplingState],
    params: DlmParameters,
    phi:    Double,
    advState: (SvdState, Double) => SvdState) = {

    val mod = Dlm.autoregressive(phi)

    // sample the T indices of the mixture
    val kt = sampleKt(ys, alphas)

    // construct a list of variances and means
    val vkt = kt.map(j => variances(j))
    val mkt = kt.map(j => means(j))

    val yt = (ys zip mkt)
      .map { case (d, m) =>
        d.copy(observation = DenseVector(d.observation(0).map(y => log(y * y) - m))) }

    val init = SvdFilter(advState).initialiseState(mod, params, yt)

    val sqrtW = params.w map math.sqrt

    // create vector of parameters
    val ps = vkt map { newV =>
      val sqrtVinv = 1.0 / math.sqrt(newV)
      params.copy(v = DenseMatrix(sqrtVinv), w = sqrtW)
    }

    val filtered = (ps zip yt).scanLeft(init) {
      case (s, (p, y)) => SvdFilter(advState).step(mod, p)(s, y)
    }

    Rand.always(FilterAr.sampleSvd(params.w, phi, filtered.toVector))
  }

  /**
    * Log-Likelihood of the AR(1) process
    * @param state the current value of the state
    * @param p the current value of the parameters
    * @return the log-likelihood
    */
  def arLikelihood(state: Vector[SamplingState], p: SvParameters): Double = {
    val n = state.length

    val ssa = (state.init zip state.tail)
      .map { case (at, at1) => at1.sample(0) - (p.mu + p.phi * (at.sample(0) - p.mu)) }
      .map(x => x * x)
      .sum

    -n * 0.5 * log(2 * math.Pi * p.sigmaEta * p.sigmaEta) - (1 / (2 * p.sigmaEta * p.sigmaEta)) * ssa
  }

  /**
    * Sample Phi using a Beta proposal distribution
    * @param tau a small tuning parameter for the beta proposal
    * @param lambda a tuning parameter for the beta proposal distribution
    * @param prior a prior distribution for the parameter phi
    * @return a Metropolis Hastings step sampling the value of Phi
    */
  def samplePhi(
    tau:    Double,
    lambda: Double,
    prior:  ContinuousDistr[Double],
    p:      SvParameters,
    alpha:  Vector[SamplingState]) = { 

    val proposal = (phi: Double) => {
      new Beta(lambda * phi + tau, lambda * (1 - phi) + tau)
    }

    val pos = (phi: Double) => {
      prior.logPdf(phi) + arLikelihood(alpha, p.copy(phi = phi))
    }

    MetropolisHastings.mhAccept[Double](proposal, pos) _
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
    alphas: Vector[SamplingState]): Rand[Double] = {

    val n = alphas.tail.size
    val pmu = prior.mean
    val psigma = prior.variance

    val sumStates = (alphas.tail.init, alphas.drop(2)).zipped.
      map { case (at, at1) => (at1.sample(0) - p.phi * at.sample(0)) }.reduce(_ + _)

    val prec = 1 / psigma + ((n - 1) * (1 - p.phi) * (1 - p.phi)) / p.sigmaEta * p.sigmaEta
    val mean = (pmu / psigma + ((1 - p.phi) / p.sigmaEta * p.sigmaEta) * sumStates) / prec
    val variance = 1 / prec

    Gaussian(mean, math.sqrt(variance))
  }

  /**
    * Sample sigma from the an inverse gamma distribution (sqrt)
    * @param prior the prior for the variance of the noise of the latent-state
    * @return a distribution over the system variance
    */
  def sampleSigma(
    prior:  InverseGamma,
    p:      SvParameters,
    alphas: Vector[SamplingState]) = {

    val squaredSum = (alphas.tail.init, alphas.drop(2)).zipped.
      map { case (mt, mt1) =>
          val diff = (mt1.sample(0) - p.mu) - p.phi * (mt.sample(0) - p.mu)
          (diff * diff)
      }.sum

    val shape = prior.shape + alphas.size * 0.5
    val scale = prior.scale + squaredSum * 0.5

    InverseGamma(shape, scale).map(math.sqrt)
  }

  /**
    * Transform the observations y to log(y^2) and remove the mean
    * @param ys a vector of univariate observations
    * @return a transformed vector of observations suitable for Kalman Filtering
    */
  def transformObservations(ys: Vector[Data]) = {
    for {
      Data(time, yo) <- ys
      yt = yo(0) map (y => log(y * y) + 1.27)
    } yield Data(time, DenseVector(yt))
  }

  /**
    * Sample the initial state from the Gaussian approximation of the SV model
    * log(y_t^2) = a_t + log(e_t^2)
    * a_t = phi a_t-1 + eta_t
    * specifying the latent state evolution
    * @return the latent-state a_t under the assumption that log(e_t^2) is Gaussian
    */
  def initialState(
    ys:       Vector[Data],
    p:        DlmParameters,
    phi:      Double,
    advState: (KfState, Double) => KfState,
    backStep: (KfState, SamplingState) => SamplingState
  ): Rand[Vector[SamplingState]] = {

    val mod = Dlm.autoregressive(phi)
    val yt = transformObservations(ys)
    val params = p.copy(v = DenseMatrix(math.Pi * math.Pi * 0.5))

    Smoothing.ffbs(mod, yt, advState, backStep, params)
  }

  /**
    * 
    */
  def stepAr(
    priorPhi:   ContinuousDistr[Double],
    priorMu:    Gaussian,
    priorSigma: InverseGamma,
    ys:         Vector[Data])(s: StochVolState) = {

    for {
      alphas <- sampleState(ys, ar1DlmParams(s.params), s.params.phi,
        FilterAr.advanceState(s.params), FilterAr.backStep(s.params))(s.alphas)
      (newPhi, accepted) <- samplePhi(0.05, 100, priorPhi, s.params, alphas)(s.params.phi)
      newSigma <- sampleSigma(priorSigma, s.params.copy(phi = newPhi), alphas)
      newMu <- sampleMu(priorMu, s.params.copy(phi = newPhi, sigmaEta = newSigma), alphas)
      p = SvParameters(newPhi, newMu, newSigma)
    } yield StochVolState(p, alphas, s.accepted + accepted)
  }

  def sampleAr(
    priorPhi:   ContinuousDistr[Double],
    priorMu:    Gaussian,
    priorSigma: InverseGamma,
    params:     SvParameters,
    ys:         Vector[Data]) = {

    // initialise the latent state
    val p = ar1DlmParams(params)
    val mod = Dlm.autoregressive(params.phi)
    val alphas = initialState(ys, p, params.phi,
      FilterAr.advanceState(params), Smoothing.step(mod, p.w)).draw
    val init = StochVolState(params, alphas, 0)

    MarkovChain(init)(stepAr(priorPhi, priorMu, priorSigma, ys))
  }

  // def stepArSvd(
  //   priorSigma: InverseGamma,
  //   priorPhi:   ContinuousDistr[Double],
  //   priorMu:    Gaussian,
  //   ys:         Vector[Data])(s: StochVolState) = {

  //   for {
  //     alphas <- sampleStateSvd(ys, s.alphas, ar1DlmParams(s.params), s.params.phi,
  //       FilterAr.advanceStateSvd(s.params))
  //     (newPhi, accepted) <- samplePhi(0.05, 100, priorPhi, s.params, alphas)(s.params.phi)
  //     newSigma <- sampleSigma(priorSigma, s.params.copy(phi = newPhi), alphas)
  //     newMu <- sampleMu(priorMu, s.params.copy(phi = newPhi, sigmaEta = newSigma), alphas)
  //     p = SvParameters(newPhi, newMu, newSigma)
  //   } yield StochVolState(p, alphas, s.accepted + accepted)
  // }

  // def sampleArSvd(
  //   priorPhi:   ContinuousDistr[Double],
  //   priorMu:    Gaussian,
  //   priorSigma: InverseGamma,
  //   params:     SvParameters,
  //   ys:         Vector[Data]) = {

  //   // initialise the latent state
  //   val p = ar1DlmParams(params)
  //   val mod = Dlm.autoregressive(params.phi)
  //   val alphas = initialState(ys, p, params.phi,
  //     FilterAr.advanceState(params),
  //     Smoothing.step(mod, p.w)).draw
  //   val init = StochVolState(params, alphas, 0)

  //   MarkovChain(init)(stepArSvd(priorPhi, priorMu, priorSigma, ys))
  // }

  def ouDlmParams(params: SvParameters): DlmParameters = {
    val c0 = math.pow(params.sigmaEta, 2) / (2 * params.phi)

    DlmParameters(
      v = DenseMatrix(1.0),
      w = DenseMatrix(params.sigmaEta * params.sigmaEta),
      m0 = DenseVector(params.mu),
      c0 = DenseMatrix(c0)
    )
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
    alphas: Vector[SamplingState]): Double = {

    val phi = (dt: Double) => exp(-p.phi * dt)
    val variance = (dt: Double) =>
      (math.pow(p.sigmaEta, 2) / (2*p.phi)) * (1 - exp(-2*p.phi*dt))

    (alphas.init zip alphas.tail).map { case (a0, a1) =>
      val dt = a1.time - a0.time
      if (dt == 0.0) {
        0.0
      } else {
        val mean = p.mu + phi(dt) * (a0.sample(0) - p.mu)
        Gaussian(mean, math.sqrt(variance(dt))).logPdf(a1.sample(0))
      }
    }.sum
  }

  /**
    * Sample the mean from the OU process using a metropolis step
    * @param prior a prior distribution for the mean parameter, mu
    * @param delta the standard deviation of the (Gaussian) proposal distribution
    * @return 
    */
  def sampleMuOu(
    prior:  ContinuousDistr[Double],
    delta:  Double = 0.05,
    p:      SvParameters,
    alphas: Vector[SamplingState]) = {

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
    alphas: Vector[SamplingState]) = {

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
    * Sample the autoregressive parameter (between 0 and 1)
    */
  def samplePhiOu(
    prior:  ContinuousDistr[Double],
    p:      SvParameters,
    alphas: Vector[SamplingState],
    lambda: Double = 10.0,
    tau:    Double = 0.05) = {

    val proposal = (phi: Double) =>
      new Beta(lambda * phi + tau, lambda * (1 - phi) + tau)

    val pos = (newPhi: Double) => {
      val newP = p.copy(phi = newPhi)
      prior.logPdf(newPhi) + ouLikelihood(newP, alphas)
    }

    MetropolisHastings.mhAccept[Double](proposal, pos) _
  }

  /**
    * Perform a single step of the MCMC Kernel for the SV with OU latent state
    * Using independent Metropolis-Hastings moves
    * @param
    */
  def stepOu(
    priorSigma: InverseGamma,
    priorPhi:   ContinuousDistr[Double],
    priorMu:    ContinuousDistr[Double],
    ys:         Vector[Data])(s: StochVolState) = {

    for {
      alphas <- sampleState(ys, ouDlmParams(s.params), s.params.phi,
        FilterOu.advanceState(s.params), FilterOu.backwardStep(s.params))(s.alphas)
      (newPhi, accepted) <- samplePhiOu(priorPhi, s.params, alphas, 0.05, 10)(s.params.phi)
      newSigma <- sampleSigmaMetropOu(priorSigma, 0.05,
        s.params.copy(phi = newPhi), alphas)(s.params.sigmaEta)
      newMu <- sampleMuOu(priorMu, 0.05,
        s.params.copy(phi = newPhi, sigmaEta = newSigma), alphas)(s.params.mu)
      p = SvParameters(newPhi, newMu, newSigma)
    } yield StochVolState(p, alphas, accepted)

  }

  def sampleOu(
    priorSigma: InverseGamma,
    priorPhi:   ContinuousDistr[Double],
    priorMu:    ContinuousDistr[Double],
    params:     SvParameters,
    ys:         Vector[Data]) = {

    // initialise the latent state
    val p = ouDlmParams(params)
    val alphas = initialState(ys, p, params.phi,
      FilterOu.advanceState(params), FilterOu.backwardStep(params)).draw
    val init = StochVolState(params, alphas, 0)

    MarkovChain(init)(stepOu(priorSigma, priorPhi, priorMu, ys))
  }
}
