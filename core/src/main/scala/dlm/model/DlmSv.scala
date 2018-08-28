package core.dlm.model

import breeze.linalg.{DenseVector, diag}
import breeze.stats.distributions._
// import breeze.numerics.{log, exp}
import cats.implicits._

/**
  * Model a heteroskedastic time series DLM by modelling the log-variance
  * of the observations as a latent-state
  */
object DlmSv {
  case class Parameters(dlm: DlmParameters,
                        sv: Vector[SvParameters])

  /**
    * Simulate a single step in the DLM with time varying observation variance
    * @param time the time of the next observation
    * @param x the state of the DLM
    * @param a latent state of the time varying log-variance
    * @param dlm the DLM model to use for the evolution
    * @param p the parameters of the DLM and FSV Model
    * @param dt the time difference between successive observations
    * @return the next simulated value
    */
  def simStep(time: Double,
              x: DenseVector[Double],
              a: Vector[Double],
              dlm: DlmModel,
              p: Parameters) = {

    for {
      wt <- MultivariateGaussian(DenseVector.zeros[Double](p.dlm.w.cols),
                                 p.dlm.w)
      at <- (a zip p.sv) traverse {
        case (ai, pi) =>
          StochasticVolatility.stepState(pi, ai, 1.0): Rand[Double]
      }
      vt <- at traverse (StochasticVolatility.observation)
      xt = dlm.g(1.0) * x + wt
      y = dlm.f(time).t * xt + DenseVector(vt.toArray)
    } yield (Dlm.Data(time, y.map(_.some)), xt, at)
  }

  /**
    * Simulate from a DLM with time varying variance represented by
    * a Stochastic Volatility Latent-State
    * @param dlm the DLM model
    * @param params the parameters of the DLM and stochastic volatility model
    * @param p the dimension of the observations
    * @return a Markov chain representing DLM with time evolving mean
    */
  def simulate(dlm: DlmModel, params: Parameters, p: Int) = {

    val initState = MultivariateGaussian(params.dlm.m0, params.dlm.c0).draw
    val initAt = params.sv.map(x => Gaussian(0.0, math.sqrt(x.sigmaEta)).draw)
    val init =
      (Dlm.Data(0.0, DenseVector[Option[Double]](None)), initState, initAt)

    MarkovChain(init) {
      case (d, x, a) => simStep(d.time + 1.0, x, a, dlm, params)
    }
  }

  /**
    * Extract a single state from a vector of states
    * @param vs the combined state
    * @param i the position of the state to extract
    * @return the extracted state
    */
  def extractState(vs: Vector[(Double, DenseVector[Double])],
                   i: Int): Vector[(Double, Double)] = {

    vs.map { case (t, x) => (t, x(i)) }
  }

  /**
    * Combine individual states into a multivariate state
    * @param s a vector of vectors containing tuples with (time, state)
    * @return a combined vector of times to state
    */
  def combineStates(s: Vector[Vector[(Double, Double)]])
    : Vector[(Double, DenseVector[Double])] = {

    s.transpose.map(
      x =>
        (
          x.head._1,
          DenseVector(x.map(_._2).toArray),
      ))
  }

  /**
    * Extract the ith factor from a multivariate vector of factors
    */
  def extractFactors(fs: Vector[(Double, Option[DenseVector[Double]])],
                     i: Int): Vector[(Double, Option[Double])] = {

    fs map {
      case (t, fo) =>
        (t, fo map (f => f(i)))
    }
  }

  /**
    * Calculate y_t - F_t x_t
    */
  def takeMean(dlm: DlmModel,
               thetas: Vector[(Double, DenseVector[Double])],
               ys: Vector[Dlm.Data]) = {

    for {
      (d, x) <- ys zip thetas.tail.map(_._2)
      fm = KalmanFilter.missingF(dlm.f, d.time, d.observation)
      y = KalmanFilter.flattenObs(d.observation)
    } yield (d.time, y - fm.t * x)
  }

  /**
    * Sample multiple independent log-volatility states representing the
    * time varying diagonal covariance matrix in a multivariate DLM
    * @param ys the value of the observation
    * @param
    * @param alphas the current log-volatility
    * @param params the parameters of the DLM SV model
    */
  def sampleStates(
    ys: Vector[Dlm.Data],
    thetas: Vector[(Double, DenseVector[Double])],
    mod: DlmModel,
    alphas: Vector[(Double, DenseVector[Double])],
    params: Parameters,
    advState: SvParameters => (KfState, Double) => KfState,
    backStep: SvParameters => (KfState, SamplingState) => SamplingState) = {

    val times = alphas.map(_._1)
    val vs = takeMean(mod, thetas, ys)

    val res: Vector[Vector[Double]] = for {
      ((v, a), ps) <- vs
        .map(_._2.data.toVector.map(_.some))
        .transpose
        .zip(alphas.map(_._2.data.toVector).transpose)
        .zip(params.sv)
      dlmP = StochasticVolatility.ar1DlmParams(ps)
      a1 = StochasticVolatility.sampleState(times zip v, 
        dlmP, ps.phi, advState(ps), backStep(ps))(times zip a)
    } yield a1.draw.map(_._2)

    Rand.always(times zip res.transpose.map(x => DenseVector(x.toArray)))
  }

  /**
    * 
    */
  def sampleVolatilityParams(
      priorPhi: ContinuousDistr[Double],
      priorSigmaEta: InverseGamma,
      alphas: Vector[(Double, DenseVector[Double])],
      params: Parameters): Vector[SvParameters] = {

    val times = alphas.map(_._1)

    for {
      (as, ps) <- alphas.map(_._2.data.toVector).transpose zip params.sv
      newPhi = StochasticVolatility
        .samplePhi(0.05, 100, priorPhi, ps, times zip as)(ps.phi)
        .draw
      newSigma = StochasticVolatility
        .sampleSigma(priorSigmaEta, ps, times zip as)
        .draw
    } yield SvParameters(newPhi, 0.0, newSigma)
  }

  /**
    * Sample the latent-state of the DLM
    * @param vs the current value of the variances
    * @param ys the current
    */
  def ffbs(
    vs: Vector[(Double, DenseVector[Double])],
    ys: Vector[Dlm.Data],
    params: DlmParameters,
    advState: (SvdState, Double) => SvdState,
    mod: DlmModel) = {

    // create a list of parameters with the variance for each time in them
    val ps = vs.map { case (t, vi) => params.copy(v = diag(vi)) }

    val init = SvdFilter(advState).initialiseState(mod, params, ys)

    // fold over the list of variances and the observations
    val filtered = (ps zip ys).scanLeft(init) {
      case (s, (p, y)) => SvdFilter(advState).step(mod, p)(s, y)
    }

    val w = SvdFilter.sqrtInvSvd(params.w)
    SvdSampler.sample(mod, filtered, w)
  }

  def initialiseVariances(p: Int, n: Int) =
    for {
      t <- Vector.range(1, n)
      x = DenseVector.rand(p, InverseGamma(1.0, 0.01))
    } yield (t.toDouble, x)

  case class State(alphas: Vector[(Double, DenseVector[Double])],
                   thetas: Vector[(Double, DenseVector[Double])],
                   params: Parameters)

  /**
    * Initialise the state of the MCMC for the DLM SV model
    */
  def initialiseState(
    params: Parameters,
    ys: Vector[Dlm.Data],
    mod: DlmModel,
    advState: SvParameters => (KfState, Double) => KfState,
    backStep: SvParameters => (KfState, SamplingState) => SamplingState): Rand[State] = {

    val vs = initialiseVariances(ys.head.observation.size, ys.size + 1)

    // extract the times and states individually
    val times = ys.map(_.time)
    val yse = ys.map(_.observation.data.toVector).transpose

    for {
      alphas <- (params.sv zip yse).traverse {
        case (pi, y) =>
          val dlmP = StochasticVolatility.ar1DlmParams(pi)
          StochasticVolatility.initialState((times zip y), dlmP, pi.phi, advState(pi), backStep(pi))
      }
      theta = ffbs(vs, ys, params.dlm, SvdFilter.advanceState(params.dlm, mod.g), mod)
    } yield State(combineStates(alphas), theta, params)
  }

  /**
    * Exponentiate the log-volatilities to get the variance of the univariate DLM
    * @param alphas the log-volatility
    */
  def getVariances(alphas: Vector[(Double, DenseVector[Double])]) = {
    alphas map { case (t, x) => (t, x map math.exp) }
  }

  /**
    * Perform a single step in the MCMC sampler for the DLM SV Model
    */
  def step(
    priorW: InverseGamma,
    priorPhi: Beta,
    priorSigmaEta: InverseGamma,
    ys: Vector[Dlm.Data],
    mod: DlmModel,
    advState: SvParameters => (KfState, Double) => KfState,
    backStep: SvParameters => (KfState, SamplingState) => SamplingState
  )(s: State): Rand[State] = {

    val vs = getVariances(s.alphas)

    for {
      alphas <- sampleStates(ys, s.thetas, mod, s.alphas, s.params, advState, backStep)
      pvs = sampleVolatilityParams(priorPhi, priorSigmaEta, alphas, s.params)
      theta = ffbs(vs, ys, s.params.dlm, SvdFilter.advanceState(s.params.dlm, mod.g), mod)
      newW <- GibbsSampling.sampleSystemMatrix(priorW, theta, mod.g)
    } yield State(alphas, theta, Parameters(s.params.dlm.copy(w = newW), pvs))
  }

  /**
    * Gibbs sampling for the DLM SV Model
    * @param priorW the prior for the state evolution noise
    * @param priorPhi the prior for the mean reverting factor phi
    * @param priorSigmaEta the prior for the variance of the log-volatility
    * @param ys a vector of time series results
    * @param mod a DLM model specification
    * @param initP the parameters
    * @return a markov chain
    */
  def sample(
    priorW: InverseGamma,
    priorPhi: Beta,
    priorSigmaEta: InverseGamma,
    ys: Vector[Dlm.Data],
    mod: DlmModel,
    initP: Parameters,
    advState: SvParameters => (KfState, Double) => KfState,
    backStep: SvParameters => (KfState, SamplingState) => SamplingState): Process[State] = {

    // initialise the latent state
    val init = initialiseState(initP, ys, mod, advState, backStep).draw

    MarkovChain(init)(step(priorW, priorPhi, priorSigmaEta, ys, mod, advState, backStep))
  }
}
