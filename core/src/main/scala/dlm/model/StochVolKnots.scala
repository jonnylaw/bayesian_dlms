package dlm.core.model

import breeze.stats.distributions._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{log, exp, sqrt}
import math._

/**
  * The state for the Gibbs Sampler
  * @param params the current value of the parameters of the SV model
  * @param alphas the current value of the AR(1) state
  */
case class StochVolState(
  params: SvParameters,
  alphas: Vector[SamplingState],
  accepted: Int)

/**
  * Use a Gaussian approximation to the state space to sample the
  * stochastic volatility model with discrete regular observations 
  * and an AR(1) latent state
  */
object StochasticVolatilityKnots {
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
    alphas: Vector[Double]): Rand[Double] = {

    val pmu = prior.mean
    val psigma = prior.variance

    val sumStates = alphas.
      tail.
      map(at => (at - p.mu)).
      map(x => x * x).
      reduce(_ + _)

    val sumStates2 = (alphas.tail.init zip alphas.drop(2)).
      map { case (at1, at) => (at1 - p.mu) * (at - p.mu) }.
      reduce(_ + _)

    val prec = 1 / psigma + (1 / p.sigmaEta * p.sigmaEta) * sumStates
    val mean = (pmu / psigma + (1 / p.sigmaEta * p.sigmaEta) * sumStates2) / prec
    val variance = 1 / prec

    Gaussian(mean, sqrt(variance))
  }

  /**
    * Transform the observations y to log(y^2) and remove the mean
    * @param ys a vector of univariate observations
    * @return a transformed vector of observations suitable for Kalman Filtering
    */
  def transformObservations(ys: Vector[Data]): Vector[Data] = {
    for {
      Data(time, y) <- ys
      yt = y map ( _.map(yp => log(yp * yp) + 1.27))
    } yield Data(time, yt)
  }

  def ffbs(
    mod: Dlm,
    observations: Vector[Data],
    advState: (KfState, Double) => KfState,
    backStep: (KfState, SamplingState) => SamplingState,
    p: DlmParameters) = {

    val filtered = KalmanFilter(advState).filter(mod, observations, p)

    val initState = Smoothing.initialise(filtered)
    val sampled = filtered.scanRight(initState)(backStep).init

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
    ys: Vector[Data]): Rand[Vector[SamplingState]] = {

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
    state: Vector[Double],
    ys:    Vector[Option[Double]]): Double = {

    val n = ys.length

    val sums = (ys zip state.tail).
      map {
        case (Some(y), a) => log(y * y) + 1.27 - a
        case _ => 0.0
      }.
      map(x => x * x).
      sum

    -3 * n * 0.5 * log(math.Pi) - (1 / (math.Pi * math.Pi)) * sums
  }

  /**
    * The exact log likelihood of the observations
    * @param state the proposed state for the current block
    * @param ys to observations for the current block
    * @return The exact log likelihood of the observations
    */
  def exactLl(
    state: Vector[Double],
    ys:    Vector[Option[Double]]): Double = {

    val n = ys.length

    val sums = (ys zip state.tail).
      map {
        case (Some(y), a) => a + y * y * exp(-a)
        case _ => 0.0
      }.
      sum

    -n * 0.5 * log(2 * math.Pi) - 0.5 * sums
  }

  def toKfState(
    fs: SamplingState): KfState = 
    KfState(fs.time, fs.mean, fs.cov, fs.mean, fs.cov, None, None, 0.0)


  def conditionalFilter(
    start:    SamplingState,
    p:        SvParameters,
    ys:       Vector[Data],
    advState: SvParameters => (KfState, Double) => KfState) = {

    val mod = Dlm.autoregressive(p.phi)
    val dlmP = StochasticVolatility.ar1DlmParams(p).
      copy(v = DenseMatrix(math.Pi * math.Pi * 0.5))

    val s0 = toKfState(start)
    ys.scanLeft(s0)(KalmanFilter(advState(p)).step(mod, dlmP))
  }

  def conditionalSampler(
    end:      SamplingState,
    p:        SvParameters,
    filtered: Vector[KfState],
    backStep: SvParameters => (KfState, SamplingState) => SamplingState) = {

    val sampled = filtered.scanRight(end)(backStep(p))

    Rand.always(sampled.tail)
  }

  /**
    * Conditional Forward filtering backward sampling
    * @param start a tuple containing the time and 
    * the mean of the state at the start of the knot and the variance
    * @param end a tuple containing the time and
    * the mean of the state at the end of the knot and the variance
    * @param p stochastic volatility parameters
    * @param ys an array of data representing discrete observations
    */
  def conditionalFfbs(
    start:    SamplingState,
    end:      SamplingState,
    p:        SvParameters,
    ys:       Vector[Data],
    advState: SvParameters => (KfState, Double) => KfState,
    backStep: SvParameters => (KfState, SamplingState) => SamplingState
  ): Rand[Vector[SamplingState]] = {

    val filtered = conditionalFilter(start, p, ys, advState)
    val sampled = conditionalSampler(end, p, filtered.init, backStep)

    sampled.map(s => start +: s)
  }

  def sampleStateArray(
    ys:       Vector[Data],
    advState: SvParameters => (KfState, Double) => KfState,
    backStep: SvParameters => (KfState, SamplingState) => SamplingState,
    p:        SvParameters,
    knots:    Vector[Int]) = { state: Array[SamplingState] =>

    for (i <- knots.indices.init) {
      val selectedObs = ys.slice(knots(i), knots(i + 1))

      if (i == 0) {
        val vs = state.slice(1, knots(i + 1) + 1).toVector
        val res = sampleStart(advState, backStep)(selectedObs, p, vs.last)(vs).draw
        state.slice(1, knots(i + 1) + 1).copyToArray(res.toArray, 1)
      } else if (i == knots.size - 2) {
        val vs = state.slice(knots(i), knots(i + 1) + 1).toVector
        val res = sampleEnd(advState, backStep)(selectedObs, p, vs.head)(vs).draw
        state.slice(knots(i), knots(i + 1)).copyToArray(res.toArray, knots(i))
      } else {
        val vs = state.slice(knots(i), knots(i + 1) + 1).toVector
        val res = sampleBlock(selectedObs, p, advState, backStep)(vs).draw
        state.slice(knots(i), knots(i + 1)).copyToArray(res.toArray, knots(i))
      }
    }

    state
  }

  def transformObs(ys: Vector[Option[Double]]) = 
    for {
      yo <- ys
      yt = yo map (y => log(y * y) + 1.27)
    } yield yt

  def sampleBlockAr(
    ys: Vector[Option[Double]],
    p:  SvParameters) = {

    val transObs = transformObs(ys)

    val prop = (vs: Vector[FilterAr.SampleState]) => 
      FilterAr.conditionalFfbs(vs.head, vs.last, p, transObs)
    val ll = (vs: Vector[FilterAr.SampleState]) => {
      val state = vs.map(_.sample)
      exactLl(state, ys) - approxLl(state, ys)
    }

    MarkovChain.Kernels.metropolis(prop)(ll)
  }

  def sampleEndAr(
    ys:    Vector[Option[Double]],
    p:     SvParameters) = {

    val transObs = transformObs(ys)

    val prop = (vs: Vector[FilterAr.SampleState]) => {
      val filtered = FilterAr.conditionalFilter(vs.head, p, transObs)
      FilterAr.univariateSample(p, filtered)
    }
    val ll = (vs: Vector[FilterAr.SampleState]) => {
      val state = vs.map(_.sample)
      exactLl(state, ys) - approxLl(state, ys)
    }

    MarkovChain.Kernels.metropolis(prop)(ll)
  }

  def sampleStartAr(
    ys:  Vector[Option[Double]],
    p:   SvParameters) = {

    val transObs = transformObs(ys)

    val prop = (vs: Vector[FilterAr.SampleState]) => {
      val variances = Vector.fill(ys.size)(Pi * Pi * 0.5)
      val filtered = FilterAr.filterUnivariate(transObs, variances, p)
      FilterAr.conditionalSampler(vs.last, p, filtered)
    }
    val ll = (vs: Vector[FilterAr.SampleState]) => {
      val state = vs.map(_.sample)
      exactLl(state, ys) - approxLl(state, ys)
    }

    MarkovChain.Kernels.metropolis(prop)(ll)
  }

  def sampleStateAr(
    ys:    Vector[Option[Double]],
    p:     SvParameters,
    knots: Vector[Int]) = { state: Vector[FilterAr.SampleState] =>

    for (i <- knots.indices.init) {
      val selectedObs = ys.slice(knots(i), knots(i + 1))

      if (i == 0) {
        val vs = state.slice(1, knots(i + 1) + 1).toVector
        val res = sampleStartAr(selectedObs, p)(vs).draw
        state.slice(1, knots(i + 1) + 1).copyToArray(res.toArray, 1)
      } else if (i == knots.size - 2) {
        val vs = state.slice(knots(i), knots(i + 1) + 1).toVector
        val res = sampleEndAr(selectedObs, p)(vs).draw
        state.slice(knots(i), knots(i + 1)).copyToArray(res.toArray, knots(i))
      } else {
        val vs = state.slice(knots(i), knots(i + 1) + 1).toVector
        val res = sampleBlockAr(selectedObs, p)(vs).draw
        state.slice(knots(i), knots(i + 1)).copyToArray(res.toArray, knots(i))
      }
    }

    state
  }

  /**
    * Sample a block of the latent state
    * @param start the starting index of the latents state and observations
    */
  def sampleBlock(
    obs:      Vector[Data],
    p:        SvParameters,
    advState: SvParameters => (KfState, Double) => KfState,
    backStep: SvParameters => (KfState, SamplingState) => SamplingState) = {

    // transform the observations for use in the ffbs algorith
    val transObs = transformObservations(obs)

    val prop = (vs: Vector[SamplingState]) => 
      conditionalFfbs(vs.head, vs.last, p, transObs, advState, backStep)
    val ll = (vs: Vector[SamplingState]) => {
      val state = vs.map(_.sample(0))
      val ys = obs.map(d => d.observation(0))
      exactLl(state, ys) - approxLl(state, ys)
    }

    MarkovChain.Kernels.metropolis(prop)(ll)
  }

  def sampleStart(
    advState: SvParameters => (KfState, Double) => KfState,
    backStep: SvParameters => (KfState, SamplingState) => SamplingState)(
    ys:  Vector[Data],
    p:   SvParameters,
    end: SamplingState): Vector[SamplingState] => Rand[Vector[SamplingState]] = { vs =>

    val c0 = (p.sigmaEta * p.sigmaEta) / (1 - p.phi * p.phi)
    val init = MultivariateGaussian(DenseVector(p.mu), DenseMatrix(c0))
    val ss = SamplingState(ys.head.time - 1, init.draw,
      init.mean, init.variance, init.mean, init.variance)

    sampleBlock(ys, p, advState, backStep)(ss +: vs)
  }

  def sampleEnd(
    advState: SvParameters => (KfState, Double) => KfState,
    backStep: SvParameters => (KfState, SamplingState) => SamplingState)(
    ys:       Vector[Data],
    p:        SvParameters,
    start:    SamplingState): Vector[SamplingState] => Rand[Vector[SamplingState]] = {

    val transObs = transformObservations(ys)

    val prop = (vs: Vector[SamplingState]) => {
      val filtered = conditionalFilter(vs.head, p, transObs, advState)
      val initState = Smoothing.initialise(filtered)
      Rand.always(filtered.init.scanRight(initState)(backStep(p)))
    }
    val ll = (vs: Vector[SamplingState]) => {
      val state = vs.map(_.sample(0))
      val obs = ys.map(_.observation(0))
      exactLl(state, obs) - approxLl(state, obs)
    }

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
    ys:       Vector[Data],
    advState: SvParameters => (KfState, Double) => KfState,
    backStep: SvParameters => (KfState, SamplingState) => SamplingState,
    p:        SvParameters,
    knots:    Vector[Int]) = { s: Vector[SamplingState] =>

    val starts = knots.init
    val ends = knots.tail
    val theEnd = knots.last

    (starts zip ends).foldLeft(s) { case (st, (start, end)) =>
      val selectedObs = ys.slice(start, end)

      val newBlock = if (start == 0) {
        val vs = st.slice(start + 1, end + 1)
        sampleStart(advState, backStep)(selectedObs, p, vs.last)(vs).draw
      } else if (end == theEnd) {
        val vs = st.slice(start, end + 1)
        sampleEnd(advState, backStep)(selectedObs, p, vs.head)(vs).draw
      } else {
        val vs = st.slice(start, end + 1)
        sampleBlock(selectedObs, p, advState, backStep)(vs).draw
      }

      st.take(start) ++ newBlock ++ st.drop(end + 1)
    }
  }

  def discreteUniform(min: Int, max: Int) =
    min + scala.util.Random.nextInt(max - min + 1)

  def sampleStarts(min: Int, max: Int)(length: Int) = {
    Stream.continually(discreteUniform(min, max)).
      scanLeft(0)(_ + _).
      takeWhile(_ < length - 1).
      toVector
  }

  /**
    * Sample knot positions by sampling block size 
    * from a uniform distribution between
    * min and max for a sequence of observations of length n
    * @param min the minimum size of a block
    * @param max the maxiumum size of a block
    * @param n the length of the observations
    */
  def sampleKnots(min: Int, max: Int)(n: Int): Rand[Vector[Int]] = 
    Rand.always(sampleStarts(min, max)(n) :+ n - 1)

  case class StochVolStateUni(
    params:   SvParameters,
    alphas:   Vector[FilterAr.SampleState],
    accepted: Int)

  def sampleStepUni(
    priorPhi:      Gaussian,
    priorMu:       Gaussian,
    priorSigmaEta: InverseGamma,
    ys:            Vector[Option[Double]]) = { st: StochVolStateUni =>
    for {
      knots <- sampleKnots(10, 100)(ys.size)
      alphas = sampleStateAr(ys, st.params, knots)(st.alphas)
      state = alphas.map(_.sample)
      phi <- samplePhi(priorPhi, st.params, state)
      mu <- StochasticVolatility.sampleMu(priorMu,
        st.params.copy(phi = phi), state)
      se <- StochasticVolatility.sampleSigma(priorSigmaEta,
        st.params.copy(phi = phi, mu = mu), state)
    } yield StochVolStateUni(SvParameters(phi, mu, se), alphas, 0)
  }

  def initialStateAr(
    p:  SvParameters,
    ys: Vector[Option[Double]]): Rand[Vector[FilterAr.SampleState]] = {

    // transform the observations, centering, squaring and logging
    val transObs = transformObs(ys)
    val vs = Vector.fill(ys.size)(Pi * Pi * 0.5)
    FilterAr.ffbs(p, transObs, vs)
  }

  def sampleUni(
    priorPhi:      Gaussian,
    priorMu:       Gaussian,
    priorSigmaEta: InverseGamma,
    ys:            Vector[Option[Double]],
    initP:         SvParameters) = {

    val init = StochVolStateUni(initP, initialStateAr(initP, ys).draw, 0)
    MarkovChain(init)(sampleStepUni(priorPhi, priorMu, priorSigmaEta, ys))
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
    priorPhi:      Gaussian,
    priorMu:       Gaussian,
    priorSigmaEta: InverseGamma,
    ys:            Vector[Data]) = { st: StochVolState =>
    for {
      knots <- sampleKnots(10, 100)(ys.size)
      alphas = sampleStateArray(ys, FilterAr.advanceState,
        FilterAr.backStep, st.params, knots)(st.alphas.toArray).toVector
      state = alphas.map(_.sample(0))
      phi <- samplePhi(priorPhi, st.params, state)
      mu <- StochasticVolatility.sampleMu(priorMu,
        st.params.copy(phi = phi), state)
      se <- StochasticVolatility.sampleSigma(priorSigmaEta,
        st.params.copy(phi = phi, mu = mu), state)
    } yield StochVolState(SvParameters(phi, mu, se), alphas, 0)
  }

  /**
    * Perform Gibbs Sampling using Conjugate specification
    * @param priorMu
    * @param priorSigmaEta Inverse Gamma prior for the variance of the noise term in the latent-state
    * @param priorPhi Gaussian prior for the parameter phi
    * @param ys a vector of observations of the time series
    * @param sampleK a function from the length of the state remaining to the new K value
    */
  def sample(
    priorPhi:      Gaussian,
    priorMu:       Gaussian,
    priorSigmaEta: InverseGamma,
    ys:            Vector[Data],
    initP:         SvParameters): Process[StochVolState] = {

    val init = StochVolState(initP, initialState(initP, ys).draw, 0)
    MarkovChain(init)(sampleStep(priorPhi, priorMu, priorSigmaEta, ys))
  }

  def sampleStepBeta(
    priorPhi:      ContinuousDistr[Double],
    priorMu:       Gaussian,
    priorSigmaEta: InverseGamma,
    ys:            Vector[Data]) = { st: StochVolState =>
    for {
      knots <- sampleKnots(10, 100)(ys.size)
      alphas = sampleStateArray(ys, FilterAr.advanceState,
        FilterAr.backStep, st.params, knots)(st.alphas.toArray).toVector
      state = alphas.map(_.sample(0))
      (phi, accepted) <- StochasticVolatility.samplePhi(0.05, 100.0,
        priorPhi, st.params, state)(st.params.phi)
      mu <- StochasticVolatility.sampleMu(priorMu,
        st.params.copy(phi = phi), state)
      se <- StochasticVolatility.sampleSigma(priorSigmaEta,
        st.params.copy(phi = phi, mu = mu), state)    
    } yield StochVolState(SvParameters(phi, mu, se), alphas, st.accepted + accepted)
  }


  def sampleBeta(
    priorPhi:      ContinuousDistr[Double],
    priorMu:       Gaussian,
    priorSigmaEta: InverseGamma,
    observations:  Vector[Data],
    initP:         SvParameters): Process[StochVolState] = {

    val init = StochVolState(initP, initialState(initP, observations).draw, 0)
    MarkovChain(init)(sampleStepBeta(priorPhi, priorMu, priorSigmaEta, observations))
  }
}
