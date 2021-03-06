package com.github.jonnylaw.dlm

import breeze.linalg.{DenseVector, DenseMatrix, diag}
import breeze.stats.distributions._
import breeze.stats.mean
import breeze.numerics.exp
import cats.Semigroup
import cats.implicits._

/**
  * Parameters of a DLM with a Factor structure for the
  * observation matrix
  * @param dlm the parameters of the (multivariate) DLM
  * @param fsv the parameters of the
  */
case class DlmFsvParameters(dlm: DlmParameters, fsv: FsvParameters) {

  def map(f: Double => Double) =
    DlmFsvParameters(dlm.map(f), fsv.map(f))

  def add(p: DlmFsvParameters) =
    DlmFsvParameters(p.dlm add dlm, p.fsv add fsv)

  def diagonal(m: DenseMatrix[Double]): DenseVector[Double] = {
    val ms = for {
      i <- 0 until m.rows
    } yield m(i, i)

    DenseVector(ms.toArray)
  }

  def toList = dlm.toList ::: fsv.toList
}

object DlmFsvParameters {
  def empty(vDim: Int, wDim: Int, p: Int, k: Int): DlmFsvParameters =
    DlmFsvParameters(
      DlmParameters.empty(vDim, wDim),
      FsvParameters.empty(p, k)
    )

  /**
    * Parse DLM FSV parameters from a list
    * @return vDim
    */
  def fromList(vDim: Int, wDim: Int, p: Int, k: Int)(
      l: List[Double]): DlmFsvParameters =
    DlmFsvParameters(
      DlmParameters.fromList(vDim, wDim)(l.take(vDim + wDim * 3)),
      FsvParameters.fromList(p, k)(l.drop(vDim + wDim * 3))
    )

  implicit def dlmFsvSemigroup = new Semigroup[DlmFsvParameters] {
    def combine(x: DlmFsvParameters, y: DlmFsvParameters) =
      x add y
  }
}

/**
  * Model a heteroskedastic time series DLM by modelling
  * the log-covariance of the observation variance
  * as latent-factors
  */
object DlmFsv {
  /**
    * Simulate a single step in the DLM FSV model
    * @param time the time of the next observation
    * @param x the state of the DLM
    * @param a the state of the factor (latent state of the time varying variance)
    * @param dlm the DLM model to use for the evolution
    * @param mod the stochastic volatility model
    * @param p the parameters of the DLM and FSV Model
    * @param dt the time difference between successive observations
    * @return the next simulated value
    */
  def simStep(time: Double,
              x: DenseVector[Double],
              a: Vector[Double],
              dlm: Dlm,
              p: DlmFsvParameters) = {
    for {
      wt <- MultivariateGaussian(DenseVector.zeros[Double](p.dlm.w.cols),
                                 p.dlm.w)
      (v, f1, a1) <- FactorSv.simStep(time, p.fsv)(a)
      vt = KalmanFilter.flattenObs(v.observation)
      x1 = dlm.g(1.0) * x + wt
      y = dlm.f(time).t * x1 + vt
    } yield (Data(time, y.map(_.some)), x1, a1)
  }

  /**
    * Simulate observations given realisations of the dlm state
    * and log-volatility of the factors
    * @param as the log-volatility
    * @param xs the state of the DLM
    * @param dlm a dlm model
    * @param p parameters of the DLM FSV model
    * @return
    */
  def obsVolatility(as: Vector[(Double, DenseVector[Double])],
                    xs: Vector[(Double, DenseVector[Double])],
                    dlm: Dlm,
                    p: DlmFsvParameters) = {

    for {
      (a, x) <- as zip xs
      f = exp(a._2 * 0.5)
    } yield (a._1, dlm.f(a._1).t * x._2)
  }

  /**
    * The observation model of the DLM FSV given the factors and the state
    * @param fs sampled factors
    * @param theta the state of the dlm
    * @param dlm the dlm model to use
    * @return a vector of observations
    */
  def observation(
      fs: Vector[(Double, Option[DenseVector[Double]])],
      theta: Vector[(Double, DenseVector[Double])],
      dlm: Dlm,
      p: DlmFsvParameters): Vector[(Double, Option[DenseVector[Double]])] = {

    for {
      (factor, x) <- fs zip theta
      obs = factor._2 map { f =>
        dlm.f(factor._1).t * x._2 + p.fsv.beta * f
      }
    } yield (factor._1, obs)
  }

  /**
    * Simulate from a DLM Factor Stochastic Volatility Model
    * @param dlm the dlm model
    * @param sv the stochastic volatility model
    * @param p dlm fsv model parameters
    * @param dt the time increment between observations
    * @return a vector of observations
    */
  def simulate(dlm: Dlm, p: DlmFsvParameters) = {
    val k = p.fsv.beta.cols
    val initState = MultivariateGaussian(p.dlm.m0, p.dlm.c0).draw
    val initFsv = Vector.fill(k)(Gaussian(0.0, 1.0).draw)
    val init =
      (Data(0.0, DenseVector[Option[Double]](None)), initState, initFsv)

    MarkovChain(init) { case (d, x, a) => simStep(d.time + 1.0, x, a, dlm, p) }
  }

  /**
    * The state of the Gibbs Sampler
    * @param p the current parameters of the MCMC
    * @param theta the current state of the mean latent state (DLM state)
    * of the DLM FSV model
    * @param factors the current state of the latent factors of the volatility
    * @param volatility the current state of the time varying variance of
    * the observations
    */
  case class State(
      p: DlmFsvParameters,
      theta: Vector[SamplingState],
      factors: Vector[(Double, Option[DenseVector[Double]])],
      volatility: Vector[SamplingState]
  )

  /**
    * Center the observations to taking away the dynamic mean of the series
    * @param observations a vector of observations
    * @param theta the state representing the evolving mean of the process
    * @param f the observation matrix: a function from time to a dense matrix
    * @return a vector containing the difference between the observations and dynamic mean
    */
  def factorObs(ys: Vector[Data],
                theta: Vector[SamplingState],
                f: Double => DenseMatrix[Double]) = {

    for {
      (y, x) <- ys.map(_.observation) zip theta.tail
      mean = f(x.time).t * x.sample
      diff = y.data.zipWithIndex.map {
        case (Some(yi), i) => Some(yi - mean(i))
        case (None, _)     => None
      }
    } yield Data(x.time, DenseVector(diff))
  }

  /**
    * Transform the state of this sampler into the state for the FSV model
    */
  def buildFactorState(s: State): FactorSv.State = {
    FactorSv.State(s.p.fsv, s.factors, s.volatility)
  }

  /**
    * Transform the state of this sampler into the state for the DLM
    */
  def buildDlmState(s: State): GibbsSampling.State =
    GibbsSampling.State(s.p.dlm, s.theta)

  /**
    * Perform forward filtering backward sampling using a
    * time dependent observation variance and the SVD Filter
    * @param model a DLM model
    * @param ys the time series of observations
    * @param p DLM parameters containing sqrtW for SVD filter / sampler
    * @param vs a vector containing V_t the time dependent variances
    */
  def ffbsSvd(model: Dlm,
              ys: Vector[Data],
              p: DlmParameters,
              vs: Vector[DenseMatrix[Double]]) = {

    val ps = vs.map(vi => SvdFilter.transformParams(p.copy(v = vi)))

    val filterStep = (params: DlmParameters) => {
      val advState = SvdFilter.advanceState(params, model.g) _
      SvdFilter(advState).step(model, params) _
    }
    val initFilter = SvdFilter(SvdFilter.advanceState(p, model.g))
      .initialiseState(model, p, ys)
    val filtered = (ps, ys).zipped
      .scanLeft(initFilter) {
        case (s, (params, y)) =>
          filterStep(params)(s, y)
      }
      .toVector

    Rand.always(SvdSampler.sample(model, filtered, ps.head.w))
  }

  /**
    * Perform a single step of the Gibbs Sampling algorithm
    * for the DLM FSV model
    */
  def sampleStep(priorBeta: Gaussian,
                 priorSigmaEta: InverseGamma,
                 priorPhi: Gaussian,
                 priorMu: Gaussian,
                 priorSigma: InverseGamma,
                 priorW: InverseGamma,
                 observations: Vector[Data],
                 dlm: Dlm,
                 p: Int,
                 k: Int)(s: State): Rand[State] = {

    val fs = buildFactorState(s)

    for {
      fs1 <- FactorSv.sampleStep(priorBeta,
                                 priorSigmaEta,
                                 priorMu,
                                 priorPhi,
                                 priorSigma,
                                 factorObs(observations, s.theta, dlm.f),
                                 p,
                                 k)(fs)
      vs = DlmFsvSystem.calculateVariance(fs1.volatility.tail,
                                          fs1.params.beta,
                                          fs1.params.v)
      theta <- ffbsSvd(dlm, observations, s.p.dlm, vs)
      state = theta.map(s => (s.time, s.sample))
      newW <- GibbsSampling.sampleSystemMatrix(priorW, state, dlm.g)
      newP = DlmFsvParameters(s.p.dlm.copy(w = newW), fs1.params)
    } yield State(newP, theta.toVector, fs1.factors, fs1.volatility)
  }

  def initialiseState(dlm: Dlm,
                      ys: Vector[Data],
                      params: DlmFsvParameters,
                      p: Int,
                      k: Int): State = {

    // initialise the variances and latent-states
    val vs = Vector.fill(ys.size)(DenseMatrix.eye[Double](p))
    val theta = ffbsSvd(dlm, ys, params.dlm, vs).draw
    val factorState =
      FactorSv.initialiseStateAr(params.fsv, factorObs(ys, theta, dlm.f), k)

    // calculate sqrt of W for SVD Filter
    val sqrtW = SvdFilter.sqrtSvd(params.dlm.w)

    State(params.copy(dlm = params.dlm.copy(w = sqrtW)),
          theta,
          factorState.factors,
          factorState.volatility)
  }

  /**
    * MCMC algorithm for DLM FSV with observation matrix having factor structure
    */
  def sample(priorBeta: Gaussian,
             priorSigmaEta: InverseGamma,
             priorPhi: Gaussian,
             priorMu: Gaussian,
             priorSigma: InverseGamma,
             priorW: InverseGamma,
             observations: Vector[Data],
             dlm: Dlm,
             initP: DlmFsvParameters): Process[State] = {

    // specify number of factors and dimension of the observation
    val beta = initP.fsv.beta
    val k = beta.cols
    val p = beta.rows
    val init = initialiseState(dlm, observations, initP, p, k)

    MarkovChain(init)(
      sampleStep(priorBeta,
                 priorSigmaEta,
                 priorPhi,
                 priorMu,
                 priorSigma,
                 priorW,
                 observations,
                 dlm,
                 p,
                 k))
  }

  def stepOu(priorBeta: Gaussian,
             priorSigmaEta: InverseGamma,
             priorPhi: Beta,
             priorMu: Gaussian,
             priorSigma: InverseGamma,
             priorW: InverseGamma,
             observations: Vector[Data],
             dlm: Dlm,
             p: Int,
             k: Int)(s: State): Rand[State] = {

    val fs = buildFactorState(s)

    for {
      fs1 <- FactorSv.stepOu(priorBeta,
                             priorSigmaEta,
                             priorMu,
                             priorPhi,
                             priorSigma,
                             factorObs(observations, s.theta, dlm.f),
                             p,
                             k)(fs)
      vs = DlmFsvSystem.calculateVariance(fs1.volatility.tail,
                                          fs1.params.beta,
                                          fs1.params.v)
      theta <- ffbsSvd(dlm, observations, s.p.dlm, vs)
      state = theta.map(s => (s.time, s.sample))
      newW <- GibbsSampling.sampleSystemMatrix(priorW, state, dlm.g)
      newP = DlmFsvParameters(s.p.dlm.copy(w = SvdFilter.sqrtSvd(newW)),
                              fs1.params)
    } yield State(newP, theta.toVector, fs1.factors, fs1.volatility)
  }

  def sampleOu(priorBeta: Gaussian,
               priorSigmaEta: InverseGamma,
               priorPhi: Gaussian,
               priorMu: Gaussian,
               priorSigma: InverseGamma,
               priorW: InverseGamma,
               observations: Vector[Data],
               dlm: Dlm,
               initP: DlmFsvParameters): Process[State] = {

    // specify number of factors and dimension of the observation
    val beta = initP.fsv.beta
    val k = beta.cols
    val p = beta.rows
    val init = initialiseState(dlm, observations, initP, p, k)

    MarkovChain(init)(
      sampleStep(priorBeta,
                 priorSigmaEta,
                 priorPhi,
                 priorMu,
                 priorSigma,
                 priorW,
                 observations,
                 dlm,
                 p,
                 k))
  }

  /**
    * Given a sequence of elements
    * (typically draws from a distribution)
    * with an implicit ordering
    * select credible intervals
    * @param xs a collection of elements
    * @param prob the interval to select from the sample (0, 1)
    * @return the sample corresponding to the prob credible interval
    */
  def quantile[A: Ordering](xs: Seq[A], prob: Double): A = {
    val index = math.floor(xs.length * prob).toInt
    val ordered = xs.sorted
    ordered(index)
  }

  /**
    * Calculate the mean and intervals of a single observation
    * @param obs a vector of vector of observations
    * @param q the quantile to sample for the credible intervals
    * @return a vector containing the time, mean, upper and lower credible intervals
    */
  def summariseInterpolation(obs: Vector[Vector[(Double, Option[Double])]],
                             q: Double) = {
    obs.transpose.map { x =>
      val sample = x.map(_._2).flatten
      (x.head._1, mean(sample), quantile(sample, q), quantile(sample, 1.0 - q))
    }
  }

  /**
    * Sample the factors, mean state and volatility while keeping the parameters constant
    */
  def sampleStateAr(ys: Vector[Data], dlm: Dlm, params: DlmFsvParameters) = {

    // specify number of factors and dimension of the observation
    val beta = params.fsv.beta
    val k = beta.cols
    val p = beta.rows

    // initialise the latent state
    val initFactorState = FactorSv.initialiseStateAr(params.fsv, ys, k)
    val factors = initFactorState.factors
    val vol = initFactorState.volatility
    val vs =
      DlmFsvSystem.calculateVariance(vol.tail, params.fsv.beta, params.fsv.v)
    val initDlmState = ffbsSvd(dlm, ys, params.dlm, vs).draw
    val init = State(params, initDlmState.toVector, factors, vol)

    def step(s: State) = {
      val vs = DlmFsvSystem.calculateVariance(s.volatility.tail,
                                              params.fsv.beta,
                                              params.fsv.v)
      for {
        theta <- ffbsSvd(dlm, ys, params.dlm, vs)
        fObs = factorObs(ys, s.theta, dlm.f)
        factors <- FactorSv.sampleFactors(fObs, params.fsv, s.volatility)
        vol <- FactorSv.sampleVolatilityAr(p, params.fsv, factors, s.volatility)
      } yield State(s.p, theta.toVector, factors, vol)
    }

    MarkovChain(init)(step)

  }

  /**
    * Sample the factors, mean state and volatility while keeping the parameters constant
    */
  def sampleStateOu(ys: Vector[Data], dlm: Dlm, params: DlmFsvParameters) = {

    // specify number of factors and dimension of the observation
    val beta = params.fsv.beta
    val k = beta.cols
    val p = beta.rows

    // initialise the latent state
    val initFactorState = FactorSv.initialiseStateAr(params.fsv, ys, k)
    val factors = initFactorState.factors
    val vol = initFactorState.volatility
    val vs =
      DlmFsvSystem.calculateVariance(vol.tail, params.fsv.beta, params.fsv.v)
    val initDlmState = ffbsSvd(dlm, ys, params.dlm, vs).draw
    val init = State(params, initDlmState.toVector, factors, vol)

    def step(s: State) = {
      val vs = DlmFsvSystem.calculateVariance(s.volatility.tail,
                                              params.fsv.beta,
                                              params.fsv.v)
      for {
        theta <- ffbsSvd(dlm, ys, params.dlm, vs)
        fObs = factorObs(ys, s.theta, dlm.f)
        factors <- FactorSv.sampleFactors(fObs, params.fsv, s.volatility)
        vol = FactorSv.sampleVolatilityOu(p, params.fsv, factors, s.volatility)
      } yield State(s.p, theta.toVector, factors, vol)
    }

    MarkovChain(init)(step)
  }

  /**
    * Perform a one-step forecast
    */
  def forecast(dlm: Dlm, p: DlmFsvParameters, ys: Vector[Data]) = {

    val fps = p.fsv.factorParams

    // initialise the log-volatility at the stationary solution
    val a0 = fps map { vp =>
      val initVar = math.pow(vp.sigmaEta, 2) / (1 - math.pow(vp.phi, 2))
      Gaussian(vp.mu, initVar).draw
    }

    val n = ys.size
    val times = ys.map(_.time)

    // advance volatility using the parameters
    val as = Vector.iterate(a0, n)(a =>
      (fps zip a).map {
        case (vp, at) =>
          StochasticVolatility.stepState(vp, at).draw
    })

    // add times to latent state
    val alphas = (as, times).zipped.map {
      case (a, t) =>
        SamplingState(t,
                      DenseVector(a.toArray),
                      DenseVector(a.toArray),
                      diag(DenseVector(a.toArray)),
                      DenseVector(a.toArray),
                      diag(DenseVector(a.toArray)))
    }

    // calculate the time dependent observation variance matrix
    val vs = DlmFsvSystem.calculateVariance(alphas, p.fsv.beta, p.dlm.v)

    val kf = KalmanFilter(
      KalmanFilter.advanceState(p.dlm.copy(v = vs.head), dlm.g))

    val init: KfState = kf.initialiseState(dlm, p.dlm, ys)

    // advance the state of the DLM using the time dependent system matrix
    (vs, ys).zipped.scanLeft(init) {
      case (st, (vi, y)) =>
        val dlmP = p.dlm.copy(v = vi)
        KalmanFilter(KalmanFilter.advanceState(dlmP, dlm.g))
          .step(dlm, dlmP)(st, y)
    }
  }
}
