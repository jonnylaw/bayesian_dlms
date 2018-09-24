package dlm.core.model

import breeze.linalg.{DenseVector, DenseMatrix, diag}
import breeze.stats.distributions._
import breeze.numerics.exp
import cats.Applicative
import cats.implicits._

/**
  * Model a heteroskedastic time series DLM by modelling the log-covariance
  * of the observation variance as latent-factors
  */
object DlmFsv {
  /**
    * Parameters of a DLM with a Factor structure for the observation matrix
    * @param dlm the parameters of the (multivariate) DLM
    * @param fsv the parameters of the 
    */
  case class Parameters(
    dlm: DlmParameters,
    fsv: FactorSv.Parameters) {

    def toList = diagonal(dlm.w).data.toList ::: fsv.toList
  }

  def diagonal(m: DenseMatrix[Double]): DenseVector[Double] = {
    val ms = for {
      i <- 0 until m.rows
    } yield m(i, i)

    DenseVector(ms.toArray)
  }

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
  def simStep(
    time: Double,
    x:    DenseVector[Double], 
    a:    Vector[Double], 
    dlm:  Dlm,
    p:    DlmFsv.Parameters) = {
    for {
      wt <- MultivariateGaussian(
        DenseVector.zeros[Double](p.dlm.w.cols), p.dlm.w)
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
  def obsVolatility(
    as:  Vector[(Double, DenseVector[Double])],
    xs:  Vector[(Double, DenseVector[Double])],
    dlm: Dlm,
    p:   Parameters) = {

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
    fs:    Vector[(Double, Option[DenseVector[Double]])],
    theta: Vector[(Double, DenseVector[Double])],
    dlm:   Dlm,
    p:     Parameters): Vector[(Double, Option[DenseVector[Double]])] = {

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
  def simulate(
    dlm: Dlm,
    p:   Parameters
  ) = {
    val k = p.fsv.beta.cols
    val initState = MultivariateGaussian(p.dlm.m0, p.dlm.c0).draw
    val initFsv = Vector.fill(k)(Gaussian(0.0, 1.0).draw)
    val init = (Data(0.0, DenseVector[Option[Double]](None)), initState, initFsv)

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
    p:          DlmFsv.Parameters,
    theta:      Vector[SamplingState],
    factors:    Vector[(Double, Option[DenseVector[Double]])],
    volatility: Vector[SamplingState]
  )

  /**
    * Center the observations to taking away the dynamic mean of the series
    * @param observations a vector of observations
    * @param theta the state representing the evolving mean of the process
    * @param f the observation matrix: a function from time to a dense matrix
    * @return a vector containing the difference between the observations and dynamic mean
    */
  def factorObs(
    ys:    Vector[Data],
    theta: Vector[SamplingState],
    f:     Double => DenseMatrix[Double]) = {

    for {
      (y, x) <- ys.map(_.observation) zip theta.tail
      mean = f(x.time).t * x.sample
      diff = y.data.zipWithIndex.
        map { 
          case (Some(yi), i) => Some(yi - mean(i))
          case (None, _) => None
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

  // /**
  //   * Helper function for DLM obs
  //   */
  // def dlmMinusFactors(
  //   obs:    Data,
  //   factor: (Double, Option[DenseVector[Double]]),
  //   beta:   DenseMatrix[Double]): Data = {

  //   // remove all partially missing data
  //   val ys = obs.observation.data.toVector.sequence.map { x =>
  //     DenseVector(x.toArray)
  //   }

  //   val observation = Applicative[Option].map2(factor._2, ys){
  //     (f, y) => y - beta * f }.map(_.data.toVector).sequence

  //   Data(obs.time, DenseVector(observation.toArray))
  // }

  // /**
  //   * Calculate the difference between the observations y_t and beta * f_t
  //   * @param observations a vector of observations
  //   * @param factors a vector of factors
  //   * @param beta the value of the factor loading matrix
  //   * @return a vector containing the difference between observations and 
  //   * beta * f_t
  //   */
  // def dlmObs(
  //   observations: Vector[Data],
  //   factors:      Vector[(Double, Option[DenseVector[Double]])],
  //   beta:         DenseMatrix[Double]) = {

  //   for {
  //     (y, f) <- observations zip factors
  //   } yield dlmMinusFactors(y, f, beta)
  // }

  /**
    * Perform forward filtering backward sampling using a
    * time dependent observation variance and the SVD
    * @param model a DLM model
    * @param ys the time series of observations
    * @param p DLM parameters containing sqrtW for SVD filter / sampler
    * @param vs a vector containing V_t the time dependent variances
    */
  def ffbsSvd(
    model: Dlm,
    ys:    Vector[Data],
    p:     DlmParameters,
    vs:    Vector[DenseMatrix[Double]]) = {

    val ps = vs.map(vi => p.copy(v = SvdFilter.sqrtInvSvd(vi)))

    val filterStep = (params: DlmParameters) => {
      val advState = SvdFilter.advanceState(params, model.g) _
      SvdFilter(advState).step(model, params) _
    }
    val initFilter = SvdFilter(SvdFilter.advanceState(p, model.g)).
      initialiseState(model, p, ys)
    val filtered = (ps, ys).
      zipped.
      scanLeft(initFilter){ case (s, (params, y)) =>
        filterStep(params)(s, y) }.
      toVector

    Rand.always(SvdSampler.sample(model, filtered, ps.head.w))
  }

  /**
    * Perform a single step of the Gibbs Sampling algorithm
    * for the DLM FSV model
    */
  def sampleStep(
    priorBeta:     Gaussian,
    priorSigmaEta: InverseGamma,
    priorPhi:      Beta,
    priorMu:       Gaussian,
    priorSigma:    InverseGamma,
    priorW:        InverseGamma,
    observations:  Vector[Data],
    dlm:           Dlm,
    p:             Int,
    k:             Int)(s: State): Rand[State] = {
   
    val fs = buildFactorState(s)

    for {
      fs1 <- FactorSv.sampleStep(priorBeta, priorSigmaEta, priorMu, priorPhi,
        priorSigma, factorObs(observations, s.theta, dlm.f), p, k)(fs)      
      vs = DlmFsvSystem.calculateVariance(fs1.volatility.tail,
        fs1.params.beta, diag(DenseVector.fill(p)(fs1.params.v)))
      theta <- ffbsSvd(dlm, observations, s.p.dlm, vs)
      newW <- GibbsSampling.sampleSystemMatrix(priorW, theta, dlm.g)
      newP = DlmFsv.Parameters(s.p.dlm.copy(w = SvdFilter.sqrtSvd(newW)), fs1.params)
    } yield State(newP, theta.toVector, fs1.factors, fs1.volatility)
  }

  def initialiseState(
    dlm:    Dlm,
    ys:     Vector[Data],
    params: DlmFsv.Parameters,
    p: Int,
    k: Int): State = {

    // initialise the variances and latent-states
    val vs = Vector.fill(ys.size)(DenseMatrix.eye[Double](p))
    val theta = ffbsSvd(dlm, ys, params.dlm, vs).draw
    val factorState = FactorSv.initialiseStateAr(params.fsv,
      factorObs(ys, theta, dlm.f), k)

    // calculate sqrt of W for SVD Filter
    val sqrtW = SvdFilter.sqrtSvd(params.dlm.w)

    State(params.copy(dlm = params.dlm.copy(w = sqrtW)),
      theta, factorState.factors, factorState.volatility)
  }

  /**
    * MCMC algorithm for DLM FSV with observation matrix having factor structure
    */
  def sample(
    priorBeta:     Gaussian,
    priorSigmaEta: InverseGamma,
    priorPhi:      Beta,
    priorMu:       Gaussian,
    priorSigma:    InverseGamma,
    priorW:        InverseGamma,
    observations:  Vector[Data],
    dlm:           Dlm,
    initP:         DlmFsv.Parameters): Process[State] = {

    // specify number of factors and dimension of the observation
    val beta = initP.fsv.beta
    val k = beta.cols
    val p = beta.rows
    val init = initialiseState(dlm, observations, initP, p, k)

    MarkovChain(init)(sampleStep(priorBeta, priorSigmaEta, priorPhi, priorMu,
      priorSigma, priorW, observations, dlm, p, k))
  }

  // def stepOu(
  //   priorBeta:     Gaussian,
  //   priorSigmaEta: InverseGamma,
  //   priorPhi:      Beta,
  //   priorMu:       Gaussian,
  //   priorSigma:    InverseGamma,
  //   priorW:        InverseGamma,
  //   observations:  Vector[Data],
  //   dlm:           Dlm,
  //   p:             Int,
  //   k:             Int)(s: State): Rand[State] = {
   
  //   val fs = buildFactorState(s)

  //   for {
  //     fs1 <- FactorSv.stepOu(priorBeta, priorSigmaEta, priorMu, priorPhi,
  //       priorSigma, factorObs(observations, s.theta, dlm.f), p, k)(fs)      
  //     vs = DlmFsvSystem.calculateVariance(fs1.volatility.tail,
  //       fs1.params.beta, diag(DenseVector.fill(p)(fs1.params.v)))
  //     theta <- ffbsSvd(dlm, observations, s.p.dlm, vs)
  //     newW <- GibbsSampling.sampleSystemMatrix(priorW, theta.toVector, dlm.g)
  //     newP = DlmFsv.Parameters(s.p.dlm.copy(w = SvdFilter.sqrtSvd(newW)), fs1.params)
  //   } yield State(newP, theta.toVector, fs1.factors, fs1.volatility)
  // }

  // def sampleOu(
  //   priorBeta:     Gaussian,
  //   priorSigmaEta: InverseGamma,
  //   priorPhi:      Beta,
  //   priorMu:       Gaussian,
  //   priorSigma:    InverseGamma,
  //   priorW:        InverseGamma,
  //   observations:  Vector[Data],
  //   dlm:           Dlm,
  //   initP:         DlmFsv.Parameters): Process[State] = {

  //   // specify number of factors and dimension of the observation
  //   val beta = initP.fsv.beta
  //   val k = beta.cols
  //   val p = beta.rows
  //   val init = initialiseState(dlm, observations, initP, p, k)

  //   MarkovChain(init)(sampleStep(priorBeta, priorSigmaEta, priorPhi, priorMu,
  //     priorSigma, priorW, observations, dlm, p, k))
  // }
}
