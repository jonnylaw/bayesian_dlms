package core.dlm.model

import breeze.stats.distributions._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{log, exp, sqrt}

/**
  * Use a Gaussian approximation to the state space to sample the stochastic volatility model
  * with discrete regular observations and an AR(1) latent state
  */
object StochasticVolatilityKnots {
  /**
    * The state for the Gibbs Sampler
    * @param params the current value of the parameters of the SV model
    * @param alphas the current value of the AR(1) state
    */
  case class State(
    params: SvParameters,
    alphas: Vector[LatentState])

  /**
    * A Single realisation of a state from the FFBS algorithm with mean and variance
    * @param time the current time of the realisation of the state
    * @param sample a sample from the state
    * @param mean the mean of the state
    * @param cov the variance of the state
    */
  case class LatentState(
    time:   Double,
    sample: Double,
    mean:   Double,
    cov:    Double)

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
    alphas: Vector[LatentState]): Rand[Double] = {

    val n = alphas.tail.size
    val pmu = prior.mean
    val psigma = prior.variance

    val sumStates = (alphas.init, alphas.tail).zipped.
      map { case (at1, at) => (at.sample - p.phi * at1.sample) }.reduce(_ + _)

    val prec = 1 / psigma + (n - 1) * (1 - p.phi) * (1 - p.phi) / p.sigmaEta * p.sigmaEta
    val mean = (pmu / psigma + ((1 - p.phi) / p.sigmaEta * p.sigmaEta) * sumStates) / prec
    val variance = 1 / prec

    for {
      mu <- Gaussian(mean, sqrt(variance))
    } yield mu
  }

  /**
    * Sample phi from the autoregressive state space
    * from a conjugate Gaussian distribution
    * @param prior a Gaussian prior distribution
    * @return a function from the current state
    * to the next state with a new value for 
    * phi sample from a Gaussian posterior distribution
    */
  def samplePhi(
    prior:  Gaussian,
    p:      SvParameters,
    alphas: Vector[LatentState]): Rand[Double] = {

    val pmu = prior.mean
    val psigma = prior.variance

    val sumStates = alphas.
      tail.
      map{ at => (at.sample - p.mu) }.
      map(x => x * x).
      reduce(_ + _)

    val sumStates2 = (alphas.init zip alphas.tail).
      map { case (at1, at) => (at1.sample - p.mu) * (at.sample - p.mu) }.
      reduce(_ + _)

    val prec = 1 / psigma + (1 / p.sigmaEta * p.sigmaEta) * sumStates
    val mean = (pmu / psigma + (1 / p.sigmaEta * p.sigmaEta) * sumStates2) / prec
    val variance = 1 / prec

    for {
      phi <- Gaussian(mean, sqrt(variance))
    } yield phi
  }

  /**
    * Sample sigma_eta from the square root of an inverse gamma distribution
    * @param prior the prior for the variance of the noise of the latent-state
    * @return a distribution over the system variance
    */
  def sampleSigmaEta(
    prior:  InverseGamma,
    p:      SvParameters,
    alphas: Vector[LatentState]): Rand[Double] = {

    // take the squared difference of a_t - phi * a_{t-1}
    // for t = 1 ... n and add them all up
    val squaredSum = (alphas.init zip alphas.tail).
      map { case (a0, a1) => (a1.sample - p.mu) - p.phi * (a0.sample - p.mu) }.
      map(x => x * x).
      sum

    val shape = prior.shape + (alphas.size - 1) * 0.5
    val scale = prior.scale + squaredSum * 0.5

    for {
      se <- InverseGamma(shape, scale)
    } yield sqrt(se)
  }

  /**
    * Transform the observations y to log(y^2) and remove the mean
    * @param ys a vector of univariate observations
    * @return a transformed vector of observations suitable for Kalman Filtering
    */
  def transformObservations(ys: Vector[Dlm.Data]): Vector[Dlm.Data] = {
    for {
      Dlm.Data(time, y) <- ys
      yt = y map ( _.map(yp => log(yp * yp) + 1.27))
    } yield Dlm.Data(time, yt)
  }

  /**
    * Perform forward filtering backward sampling, returning the latent
    * state
    */
  def ffbs(
    mod: DlmModel,
    observations: Vector[Dlm.Data],
    advState: (KfState, Double) => KfState,
    backStep: (KfState, SamplingState) => SamplingState,
    p: DlmParameters) = {

    val filtered = KalmanFilter(advState).filter(mod, observations, p)

    val initState = Smoothing.initialise(filtered)
    val sampled = filtered.scanRight(initState)(backStep).
      map(s => LatentState(s.time, s.sample(0), s.mean(0), s.cov(0,0)))

    Rand.always(sampled)
  }

  /**
    * Sample the initial state from the Gaussian approximation of the SV model
    * log(y_t^2) = a_t + log(e_t^2)
    * a_t = phi a_t-1 + eta_t
    * specifying the latent state evolution
    * @param p the parameters of the stochastic volatility model
    * @param obs the time series observations: y_t = e_t * exp(a_t / 2)
    * @return the latent-state a_t under the assumption that log(e_t^2) is Gaussian
    */
  def initialState(
    p:  SvParameters,
    ys: Vector[Dlm.Data]): Rand[Vector[LatentState]] = {

    // transform the observations, centering, squaring and logging
    val observations = transformObservations(ys)
    val mod = Dlm.autoregressive(p.phi)
    val params = StochasticVolatility.ar1DlmParams(p).
      copy(v = DenseMatrix(math.Pi * math.Pi * 0.5))

    ffbs(mod, observations, FilterAr.advanceState(p), FilterAr.backStep(p), params)
  }

  /**
    * The log likelihood for the Gaussian approximation 
    * @param state the proposed state for the current block
    * @param observations to observations for the current block
    * @return the log likelihood of the Gaussian approximation 
    */
  def approxLl(
    state:        Vector[LatentState],
    observations: Vector[Dlm.Data]): Double = {

    val n = observations.length

    val sums = (observations.map(_.observation(0)) zip state.tail).
      map {
        case (Some(y), a) => log(y * y) + 1.27 - a.sample
        case _ => 0.0
      }.
      map(x => x * x).
      sum

    -3 * n * 0.5 * log(math.Pi) - (1 / (math.Pi * math.Pi)) * sums
  }

  /**
    * The exact log likelihood of the observations
    * @param state the proposed state for the current block
    * @param observations to observations for the current block
    * @return The exact log likelihood of the observations
    */
  def exactLl(
    state:        Vector[LatentState],
    observations: Vector[Dlm.Data]): Double = {

    val n = observations.length

    val sums = (observations.map(_.observation(0)) zip state.tail).
      map {
        case (Some(y), a) => a.sample + y * y * exp(-a.sample)
        case _ => 0.0
      }.
      sum

    -n * 0.5 * log(2 * math.Pi) - 0.5 * sums
  }

  def toKfState(
    fs: LatentState): KfState = {

    KfState(fs.time, DenseVector(fs.mean), DenseMatrix(fs.cov),
      DenseVector(fs.mean), DenseMatrix(fs.cov), None, None, 0.0)
  }

  /**
    * Conditional Forward filtering backward sampling
    * @param start a tuple containing the time and 
    * the mean of the state at the start of the knot and the variance
    * @param end a tuple containing the time and
    * the mean of the state at the end of the knot and the variance
    * @param p stochastic volatility parameters
    * @param observations an array of data representing discrete observations
    */
  def conditionalFfbs(
    start:    LatentState,
    end:      LatentState,
    p:        SvParameters,
    ys:       Vector[Dlm.Data],
    advState: SvParameters => (KfState, Double) => KfState,
    backStep: SvParameters => (KfState, SamplingState) => SamplingState
  ): Rand[Vector[LatentState]] = {

    // perform the Kalman Filter (conditional on the start of the knot)
    val s0 = toKfState(start)
    val sk = toKfState(end)

    val mod = Dlm.autoregressive(p.phi)
    val dlmP = StochasticVolatility.ar1DlmParams(p).
      copy(v = DenseMatrix(math.Pi * math.Pi * 0.5))

    val filtered = ys.scanLeft(s0)(KalmanFilter(advState(p)).step(mod, dlmP)).init :+ sk

    val initState = Smoothing.initialise(filtered)
    val sampled = filtered.scanRight(initState)(backStep(p)).init.tail.
      map(s => LatentState(s.time, s.sample(0), s.mean(0), s.cov(0,0)))

    Rand.always(start +: sampled :+ end)
  }


  /**
    * Sample a block of the latent state
    * @param start the starting index of the latents state and observations
    */
  def sampleBlock(
    obs:      Vector[Dlm.Data],
    p:        SvParameters,
    advState: SvParameters => (KfState, Double) => KfState,
    backStep: SvParameters => (KfState, SamplingState) => SamplingState) = {

    // transform the observations for use in the ffbs algorithm
    val transObs = transformObservations(obs)

    val prop = (vs: Vector[LatentState]) => {
      conditionalFfbs(vs.head, vs.last, p, transObs, advState, backStep)
    }
    val ll = (vs: Vector[LatentState]) =>
      exactLl(vs, obs) - approxLl(vs, obs)

    MarkovChain.Kernels.metropolis(prop)(ll)
  }

  /**
    * Sample the state in blocks
    * @param ys the full sequence of observations
    * @param advState
    * @param backStep
    * @param knots 
    * @return a function from the current state of the Markov chain
    * to the next state containing a new proposed value of the latent AR(1) state
    */
  def sampleState(
    ys:       Vector[Dlm.Data],
    advState: SvParameters => (KfState, Double) => KfState,
    backStep: SvParameters => (KfState, SamplingState) => SamplingState,
    knots:   Vector[Int]) = { s: State =>

    val starts = knots.init
    val ends = knots.tail

    (starts zip ends).foldLeft(s.alphas) { (st, indices) =>
      val (start, end) = indices
      val selectedObs = ys.slice(start, end)
      val vs = st.slice(start, end + 1)
      val newBlock = sampleBlock(selectedObs, s.params, advState, backStep)(vs).draw

      st.take(start) ++ newBlock ++ st.drop(end)
    }
  }

  /**
    * Sample from the discrete uniform distribution
    * @param min
    */
  def sampleLength(min: Int, max: Int): Rand[Int] =
    Uniform(min, max).map(u => math.floor(u).toInt)

  /**
    * Sample a vector of starts
    */
  def sampleStarts(min: Int, max: Int)(length: Int): Vector[Int] = {
    Stream.continually(sampleLength(min, max).draw).
      scanLeft(0)(_ + _).
      takeWhile(_ < length - 1).
      toVector
  }

  /**
    * Sample knot positions by sampling block size from a uniform distribution between
    *  min and max for a sequence of observations of length n
    * @param min the minimum size of a block
    * @param max the maxiumum size of a block
    * @param n the length of the observations
    */
  def sampleKnots(min: Int, max: Int)(n: Int): Rand[Vector[Int]] = {
    Rand.always(sampleStarts(min, max)(n) :+ n - 1)
  }

  /**
    * A single step in the Gibbs Sampler
    * @param priorMu a Gaussian prior for the mean parameters
    * @param priorSigmaEta Inverse Gamma prior for the variance
    * of the noise term in the latent-state
    * @param priorPhi Gaussian prior for the autoregressive parameter phi
    * @param observations a vector of observations of the time series
    * @return a kernel function State => Rand[State]
    */
  def sampleStep(
    priorMu:       Gaussian,
    priorSigmaEta: InverseGamma,
    priorPhi:      Gaussian,
    ys:            Vector[Dlm.Data]) = { st: State =>
    for {
      knots <- sampleKnots(10, 100)(ys.size)
      alphas = sampleState(ys,
        FilterAr.advanceState, FilterAr.backStep, knots)(st)
      phi <- samplePhi(priorPhi, st.params, alphas)
      mu <- sampleMu(priorMu, st.params.copy(phi = phi), alphas)
      se <- sampleSigmaEta(priorSigmaEta, st.params.copy(phi = phi, mu = mu), alphas)
    } yield State(SvParameters(phi, mu, se), alphas)
  }

  /**
    * Perform Gibbs Sampling using Conjugate specification
    * @param priorMu
    * @param priorSigmaEta Inverse Gamma prior for the variance of the noise term in the latent-state
    * @param priorPhi Gaussian prior for the parameter phi
    * @param observations a vector of observations of the time series
    * @param sampleK a function from the length of the state remaining to the new K value
    */
  def sample(
    priorMu:       Gaussian,
    priorSigmaEta: InverseGamma,
    priorPhi:      Gaussian,
    observations:  Vector[Dlm.Data],
    initP:         SvParameters): Process[State] = {

    val init = State(initP, initialState(initP, observations).draw)
    MarkovChain(init)(sampleStep(priorMu, priorSigmaEta, priorPhi, observations))
  }
}
