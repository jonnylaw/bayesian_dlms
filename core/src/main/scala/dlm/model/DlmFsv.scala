package core.dlm.model

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.distributions._
import breeze.numerics.exp
//import cats.Applicative
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

    def toList = dlm.toList ::: fsv.toList
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
    dlm:  DlmModel,
    p:    DlmFsv.Parameters) = {
    for {
      wt <- MultivariateGaussian(
        DenseVector.zeros[Double](p.dlm.w.cols), p.dlm.w)
      (v, f1, a1) <- FactorSv.simStep(time, p.fsv)(a)
      vt = KalmanFilter.flattenObs(v.observation)
      x1 = dlm.g(1.0) * x + wt
      y = dlm.f(time).t * x1 + vt
    } yield (Dlm.Data(time, y.map(_.some)), x1, a1)
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
    dlm: DlmModel,
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
    dlm:   DlmModel,
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
    dlm: DlmModel,
    p:   Parameters
  ) = {
    val k = p.fsv.beta.cols
    val initState = MultivariateGaussian(p.dlm.m0, p.dlm.c0).draw
    val initFsv = Vector.fill(k)(Gaussian(0.0, 1.0).draw)
    val init = (Dlm.Data(0.0, DenseVector[Option[Double]](None)), initState, initFsv)

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
    theta:      Vector[(Double, DenseVector[Double])],
    factors:    Vector[(Double, Option[DenseVector[Double]])],
    volatility: Vector[(Double, DenseVector[Double])]
  )

  /**
    * Center the observations to taking away the dynamic mean of the series
    * @param observations a vector of observations
    * @param theta the state representing the evolving mean of the process
    * @param f the observation matrix: a function from time to a dense matrix
    * @return a vector containing the difference between the observations and dynamic mean
    */
  def factorObs(
    observations: Vector[Dlm.Data],
    theta:        Vector[(Double, DenseVector[Double])],
    f:            Double => DenseMatrix[Double]) = {

    for {
      (y, x) <- observations.map(_.observation) zip theta
      mean = f(x._1).t * x._2
      diff = y.data.zipWithIndex.
        map { 
          case (Some(yi), i) => Some(yi - mean(i))
          case (None, _) => None
        }
    } yield Dlm.Data(x._1, DenseVector(diff))
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
  def buildDlmState(s: State): GibbsSampling.State = {
    GibbsSampling.State(
      s.p.dlm,
      s.theta.map(x => (x._1, x._2))
    )
  }

  // /**
  //   * Helper function for DLM obs
  //   */
  // def dlmMinusFactors(
  //   obs:    Dlm.Data,
  //   factor: (Double, Option[DenseVector[Double]]),
  //   beta:   DenseMatrix[Double]): Dlm.Data = {

  //   // remove all partially missing data
  //   val ys = obs.observation.data.toVector.sequence.map { x =>
  //     DenseVector(x.toArray)
  //   }

  //   val observation = Applicative[Option].map2(factor._2, ys){
  //     (f, y) => y - beta * f }.map(_.data.toVector).sequence

  //   Dlm.Data(obs.time, DenseVector(observation.toArray))
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
  //   observations: Vector[Dlm.Data],
  //   factors:      Vector[(Double, Option[DenseVector[Double]])],
  //   beta:         DenseMatrix[Double]) = {

  //   for {
  //     (y, f) <- observations zip factors
  //   } yield dlmMinusFactors(y, f, beta)
  // }

  def sampleMeanState(
    dlm: DlmModel,
    ys:  Vector[Dlm.Data],
    p:   DlmParameters,
    vs:  Vector[DenseMatrix[Double]]): Rand[Vector[(Double, DenseVector[Double])]] = {

    val ps = vs map (newV => p.copy(v = newV))
    val init = KalmanFilter.initialiseState(dlm, p, ys)

    val filtered = (ps zip ys).scanLeft(init) {
      case (s, (p, y)) => KalmanFilter.step(dlm, p,
        KalmanFilter.advanceState(p, dlm.g))(s, y)
    }

    Rand.always(Smoothing.sampleDlm(dlm, filtered.toVector, p.w))
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
    observations:  Vector[Dlm.Data],
    dlm:           DlmModel,
    p:             Int,
    k:             Int)(s: State): Rand[State] = {
   
    val beta = s.p.fsv.beta
    val fs = buildFactorState(s)

    for {
      fs1 <- FactorSv.sampleStep(priorBeta, priorSigmaEta, priorMu, priorPhi, 
        priorSigma, factorObs(observations, s.theta, dlm.f), p, k)(fs)
      vs = fs1.volatility map { case (t, a) => beta.t * a * beta + s.p.fsv.v }
      theta <- sampleMeanState(dlm, observations, s.p.dlm, vs)
      newW <- GibbsSampling.sampleSystemMatrix(priorW, theta.toVector, dlm.g)
      newP = DlmFsv.Parameters(s.p.dlm.copy(w = newW), fs1.params)
    } yield State(newP, theta.toVector, fs1.factors, fs1.volatility)
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
    observations:  Vector[Dlm.Data],
    dlm:           DlmModel,
    initP:         DlmFsv.Parameters): Process[State] = {

    // specify number of factors and dimension of the observation
    val beta = initP.fsv.beta
    val k = beta.cols
    val p = beta.rows

    // initialise the latent state
    val initFactorState = FactorSv.initialiseStateAr(initP.fsv, observations, k)
    val vs = initFactorState.volatility map { case (t, a) => beta.t * a * beta + initP.fsv.v }
    val initDlmState = sampleMeanState(dlm, observations, initP.dlm, vs).draw
    val init = State(initP, initDlmState.toVector,
      initFactorState.factors, initFactorState.volatility)

    MarkovChain(init)(sampleStep(priorBeta, priorSigmaEta, priorPhi, priorMu,
      priorSigma, priorW, observations, dlm, p, k))
  }
}
