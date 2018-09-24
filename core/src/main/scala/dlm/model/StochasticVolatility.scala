package dlm.core.model

import breeze.linalg.{DenseVector}
import breeze.stats.distributions._
import math._
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

    val phi = exp(-p.phi * dt)
    val mean = p.mu + phi * (at - p.mu)
    val variance = (math.pow(p.sigmaEta, 2) * (1 - exp(-2*p.phi*dt))) / (2*p.phi)

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
  def sampleKt(
    ys:     Vector[Option[Double]],
    alphas: Vector[Double]) = {

    // conditional likelihood
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
      (y, x) <- ys zip alphas.tail
      lw = logWeights(y, x)
      max = lw.max
      weights = lw.map(w => exp(w - max))
      kt = Multinomial(DenseVector(weights.toArray)).draw
    } yield kt
  }

  def sampleStateAr(
    ys:     Vector[(Double, Option[Double])],
    params: SvParameters,
    alphas: Vector[FilterAr.SampleState])= {

    // sample the T indices of the mixture
    val kt = sampleKt(ys.map(_._2), alphas.map(_.sample))

    // construct a list of variances and means
    val vkt = kt.map(j => variances(j))
    val mkt = kt.map(j => means(j))

    val yt = (ys zip mkt).map { case ((t, yo), m) =>
      (t, yo map (y => log(y * y) - m)) }

    // create vector of parameters
    val filtered = FilterAr.filterUnivariate(yt, vkt, params)
    FilterAr.univariateSample(params, filtered)
  }

  /**
    * Log-Likelihood of the AR(1) process
    * @param state the current value of the state
    * @param p the current value of the parameters
    * @return the log-likelihood
    */
  def arLikelihood(
    alphas: Vector[Double],
    p: SvParameters): Double = {

    val initvar = math.pow(p.sigmaEta, 2) / (1 - math.pow(p.phi, 2))

    Gaussian(p.mu, math.sqrt(initvar)).logPdf(alphas.head) +
      (alphas.drop(2) zip alphas.tail.init).map {
        case (a1, a0) =>
          val mean = p.mu + p.phi * (a0 - p.mu)
          Gaussian(mean, p.sigmaEta).logPdf(a1)
      }.sum
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
    alpha:  Vector[Double]) = { 

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
    * a new state with a new mu sampled from the Gaussian 
    * posterior distribution
    */
  def sampleMu(
    prior:  Gaussian,
    p:      SvParameters,
    alphas: Vector[Double]): Rand[Double] = {

    val n = alphas.tail.size
    val pmu = prior.mean
    val psigma = prior.variance

    val sumStates = (alphas.tail.init, alphas.drop(2)).zipped.
      map { case (at, at1) => (at1 - p.phi * at) }.reduce(_ + _)

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
    alphas: Vector[Double]) = {

    val squaredSum = (alphas.tail.init, alphas.drop(2)).zipped.
      map { case (mt, mt1) =>
          val diff = (mt1 - p.mu) - p.phi * (mt - p.mu)
          (diff * diff)
      }.sum

    val shape = prior.shape + alphas.size * 0.5
    val scale = prior.scale + squaredSum * 0.5

    InverseGamma(shape, scale).map(math.sqrt)
  }

  def stepUni(
    priorPhi:   ContinuousDistr[Double],
    priorMu:    Gaussian,
    priorSigma: InverseGamma,
    ys:         Vector[(Double, Option[Double])]) = { s: StochVolState =>

    for {
      alphas <- sampleStateAr(ys, s.params, s.alphas)
      state = alphas.map(_.sample)
      (newPhi, accepted) <- samplePhi(0.05, 100.0, priorPhi, s.params, state)(s.params.phi)
      newSigma <- sampleSigma(priorSigma, s.params.copy(phi = newPhi), state)
      newMu <- sampleMu(priorMu, s.params.copy(phi = newPhi, sigmaEta = newSigma), state)
      p = SvParameters(newPhi, newMu, newSigma)
    } yield s.copy(params = p, alphas = alphas, accepted = s.accepted + accepted)
  }

  def sampleUni(
    priorPhi:   ContinuousDistr[Double],
    priorMu:    Gaussian,
    priorSigma: InverseGamma,
    params:     SvParameters,
    ys:         Vector[(Double, Option[Double])]) = {

    // initialise the latent state
    val initState = StochasticVolatilityKnots.initialStateAr(params, ys).draw
    val init = StochVolState(params, initState, 0)

    MarkovChain(init)(stepUni(priorPhi, priorMu, priorSigma, ys))
  }

  /**
    * Marginal log likelihood of the OU process used to
    * perform the Metropolis Hastings steps to learn the static parameters
    * @param s the current state of the MCMC including the current values
    * of the parameters
    * @return the log likelihood of the state given the static parameter values
    */
  def ouLikelihood(
    p:      SvParameters,
    alphas: Vector[(Double, Double)]): Double = {

    val phi = (dt: Double) => exp(-p.phi * dt)
    val variance = (dt: Double) =>
      (math.pow(p.sigmaEta, 2) * (1 - exp(-2*p.phi*dt))) / (2*p.phi)

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
    * @param delta the standard deviation of the (Gaussian) 
    * proposal distribution
    * @return 
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

    Metropolis.mAccept[Double](proposal, pos) _
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

    Metropolis.mAccept[Double](proposal, pos) _
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

    Metropolis.mAccept[Double](proposal, pos) _
  }

  def sampleStateOu(
    ys:     Vector[(Double, Option[Double])],
    params: SvParameters,
    alphas: Vector[FilterAr.SampleState])= {

    // sample the T indices of the mixture
    val kt = sampleKt(ys.map(_._2), alphas.map(_.sample))

    // construct a list of variances and means
    val vkt = kt.map(j => variances(j))
    val mkt = kt.map(j => means(j))

    val yt = (ys zip mkt).map { case ((t, yo), m) =>
      (t, yo map (y => log(y * y) - m)) }

    // create vector of parameters
    val filtered = FilterOu.filterUnivariate(yt, vkt, params)
    FilterOu.univariateSample(params, filtered)
  }

  /**
    * Perform a single step of the MCMC Kernel for the SV with OU latent state
    * Using independent Metropolis-Hastings moves
    * @param
    */
  def stepOu(
    priorPhi:   ContinuousDistr[Double],
    priorMu:    ContinuousDistr[Double],
    priorSigma: InverseGamma,
    ys:         Vector[(Double, Option[Double])])(s: StochVolState) = {

    for {
      alphas <- sampleStateOu(ys, s.params, s.alphas)
      state = alphas map (x => (x.time, x.sample))
      (newPhi, accepted) <- samplePhiOu(priorPhi, s.params, state,
        0.05)(s.params.phi)
      (newSigma, _) <- sampleSigmaMetropOu(priorSigma, 0.05,
        s.params.copy(phi = newPhi), state)(s.params.sigmaEta)
      (newMu, _) <- sampleMuOu(priorMu, 0.05,
        s.params.copy(phi = newPhi, sigmaEta = newSigma), state)(s.params.mu)
      p = SvParameters(newPhi, newMu, newSigma)
    } yield StochVolState(p, alphas, accepted)
  }

  def initialStateOu(
    p:  SvParameters,
    ys: Vector[(Double, Option[Double])]): Rand[Vector[FilterAr.SampleState]] = {

    // transform the observations, centering, squaring and logging
    val transObs = StochasticVolatilityKnots.transformObs(ys)
    val vs = Vector.fill(ys.size)(Pi * Pi * 0.5)
    FilterOu.ffbs(p, transObs, vs)
  }

  def sampleOu(
    priorPhi:   ContinuousDistr[Double],
    priorMu:    ContinuousDistr[Double],
    priorSigma: InverseGamma,
    params:     SvParameters,
    ys:         Vector[(Double, Option[Double])]) = {

    // initialise the latent state
    val alphas = initialStateOu(params, ys).draw
    val init = StochVolState(params, alphas, 0)

    MarkovChain(init)(stepOu(priorPhi, priorMu, priorSigma, ys))
  }
}
