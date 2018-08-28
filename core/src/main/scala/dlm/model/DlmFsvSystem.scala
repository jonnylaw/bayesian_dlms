package core.dlm.model

import breeze.linalg.{DenseVector, DenseMatrix, diag}
import breeze.stats.distributions._
import breeze.stats.mean
import cats.implicits._
import breeze.numerics.exp

/**
  * Fit a DLM with the system variance modelled using an FSV model
  * and latent log volatility modelled using continuous time Ornstein-Uhlenbeck process
  */
object DlmFsvSystem {
  /**
    * The state of the Gibbs Sampler
    * @param p the current parameters of the MCMC
    * @param theta the current state of the mean latent state (DLM state)
    * of the DLM FSV model
    * @param factors the factors of the observation model
    * @param volatility the log-volatility of the system variance
    */
  case class State(
    p:          DlmFsvSystem.Parameters,
    theta:      Vector[(Double, DenseVector[Double])],
    factors:    Vector[(Double, Option[DenseVector[Double]])],
    volatility: Vector[(Double, DenseVector[Double])]
  )

  /**
    * Parameters of the DLM Factor Stochastic Volatility model
    * @param m0 the mean of the initial state 
    * @param c0 the variance of the initial state
    * @param v the measurement variance
    * @param factors the parameters of the Factor Stochastic volatility
    * model for the measurement error 
    */
  case class Parameters(
    m0:      DenseVector[Double],
    c0:      DenseMatrix[Double],
    v:       Double,
    factors: FactorSv.Parameters) {

    def toList = List(v) ::: factors.toList
  }

  /**
    * Simulate a single step in the DLM FSV model
    * @param time the time of the next observation
    * @param x the state of the DLM
    * @param a0v the latent log-volatility of the observation variance
    * @param a0w the latent log-volatility of the system variance
    * @param dlm the DLM model to use for the evolution
    * @param mod the stochastic volatility model
    * @param p the parameters of the DLM and FSV Model
    * @param dt the time difference between successive observations
    * @return the next simulated value 
    */
  def simStep(
    time:   Double,
    x:      DenseVector[Double], 
    a0w:    Vector[Double], 
    dlm:    DlmModel,
    p:      Parameters,
    dt:     Double,
    dimObs: Int) = {

    for {
      (w, f1w, a1w) <- FactorSv.simStep(time, p.factors)(a0w)
      wt = KalmanFilter.flattenObs(w.observation)
      vt = DenseVector.rand(dimObs, Gaussian(0.0, p.v))
      x1 = dlm.g(dt) * x + wt
      y = dlm.f(time).t * x1 + vt
    } yield (Dlm.Data(time, y.map(_.some)), x1, a1w)
  }

  /**
    * Simulate from a DLM Factor Stochastic Volatility Model
    * @param dlm the dlm model
    * @param sv the stochastic volatility model
    * @param p dlm fsv model parameters
    * @param dt the time increment between observations
    * @return a vector of observations
    */
  def simulateRegular(
    dlm:    DlmModel,
    p:      Parameters,
    dimObs: Int) = {

    val k = p.factors.beta.cols

    val init = for {
      initState <- MultivariateGaussian(p.m0, p.c0)
      initFw = Vector.fill(k)(Gaussian(0.0, 1.0).draw)
    } yield (Dlm.Data(0.0, DenseVector[Option[Double]](None)),
       initState, initFw)

    MarkovChain(init.draw) { case (d, x, aw) =>
      simStep(d.time + 1.0, x, aw, dlm, p, 1.0, dimObs) }
  }

  /**
    * Center the state by taking away the dynamic mean of the series
    * @param theta the state representing the evolving mean of the process
    * @param g the system matrix: a function from time to a dense matrix
    * @return a vector containing the x(t_i) - g(dt_i)x(t_{i-1})
    */
  def factorState(
    theta: Vector[(Double, DenseVector[Double])],
    g:     Double => DenseMatrix[Double]) = {

    for {
      (x0, x1) <- theta.init zip theta.tail
      dt = x1._1 - x0._1
      diff = x1._2 - g(dt) * x0._2
    } yield Dlm.Data(x1._1, diff.mapValues(_.some))
  }

  /**
    * Sample from the full conditional distribution of the measurement variance
    * @param priorV an inverse gamma prior
    * @param f the observation matrix
    * @param theta the current sample of the state
    * @param ys observations of the process
    * @return a distribution over the measurement variance
    */
  def sampleObservationVariance(
    priorV: InverseGamma,
    f:      Double => DenseMatrix[Double],
    theta:  Vector[(Double, DenseVector[Double])],
    ys:     Vector[Dlm.Data]): Rand[Double] = {

    val ssy = (theta.tail zip ys).
      map { case ((time, x), y) => 
        val fm = KalmanFilter.missingF(f, time, y.observation)
        val yt = KalmanFilter.flattenObs(y.observation)
        (yt - fm.t * x)
      }.
      map(x => x *:* x).
      reduce(_ + _)

    val shape = priorV.shape + ys.size * 0.5
    val rate = priorV.scale + mean(ssy) * 0.5

    InverseGamma(shape, rate)
  }

  /**
    * Calculate the time dependent variance from the log-volatility
    * and factor loading matrix
    * @param alphas the time series of log-volatility
    * @param beta the factor loading matrix
    * @param v the diagonal observation covariance
    */
  def calculateVariance(
    alphas: Vector[(Double, DenseVector[Double])],
    beta:   DenseMatrix[Double],
    v:      DenseMatrix[Double]): Vector[DenseMatrix[Double]] = {

    alphas map (a => ((beta * diag(exp(a._2))) * beta.t) + v)
  }

  /**
    * Perform forward filtering backward sampling using a 
    * time dependent state covariance matrix
    */
  def ffbs(
    model: DlmModel,
    ys:    Vector[Dlm.Data],
    p:     DlmParameters,
    ws:    Vector[DenseMatrix[Double]]) = {

    val ps = ws.map(wi => SvdFilter.transformParams(p.copy(w = wi)))

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

    val init = SvdSampler.initialise(filtered.toArray)
    val sampleStep = (params: DlmParameters) => {
      SvdSampler.step(model, params.w) _
    }
    val res = (ps, filtered.init).
      zipped.
      scanRight(init){ case ((params, fs), s) =>
        sampleStep(params)(fs, s) }.
      toVector

    Rand.always(res.map(st => (st.time, st.theta)))
  }

  def toDlmParameters(
    params: DlmFsvSystem.Parameters,
    p:      Int): DlmParameters = {

    DlmParameters(
      v = diag(DenseVector.fill(p)(params.v)),
      w = DenseMatrix.eye[Double](p),
      m0 = params.m0,
      c0 = params.c0)
  }

  /**
    * Perform a single step of the Gibbs Sampling algorithm
    * for the DLM FSV where the system variance is modelled
    * using FSV model
    */
  def sampleStep(
    priorBeta:     Gaussian,
    priorSigmaEta: InverseGamma,
    priorPhi:      Beta,
    priorMu:       Gaussian,
    priorSigma:    InverseGamma,
    priorV:        InverseGamma,
    ys:            Vector[Dlm.Data],
    dlm:           DlmModel)(s: State): Rand[State] = {

    // extract the system factors
    val beta = s.p.factors.beta
    val fs = FactorSv.State(s.p.factors, s.factors, s.volatility)

    // calculate the dimensions of the system and assoc. factors
    val k = beta.cols
    val d = beta.rows

    // observation dimension
    val p = ys.head.observation.size

    for {
      fs1 <- FactorSv.sampleStep(priorBeta, priorSigmaEta, priorMu,
        priorPhi, priorSigma, factorState(s.theta, dlm.g), d, k)(fs)

      // perform FFBS using time dependent system noise covariance
      ws = calculateVariance(fs1.volatility.tail, fs1.params.beta,
        diag(DenseVector.fill(d)(fs1.params.v)))
      dlmP = toDlmParameters(s.p, p)
      theta <- ffbs(dlm, ys, dlmP, ws)

      // newV <- sampleObservationVariance(priorV, dlm.f, theta, ys)
      newP = s.p.copy(factors = fs1.params)
        //, v = newV)
    } yield State(newP, theta, fs1.factors, fs1.volatility)
  }

  /**
    * Initialise the state of the DLM FSV system Model
    * by initialising variance matrices for the system, performing FFBS for
    * the mean state
    * @param params parameters of the DLM FSV system model
    * @param ys time series of observations
    * @param dlm the description of the 
    */
  def initialise(
    params: DlmFsvSystem.Parameters,
    ys:     Vector[Dlm.Data],
    dlm:    DlmModel) = {

    val k = params.factors.beta.cols
    val p = ys.head.observation.size
    val parameters = toDlmParameters(params, p)

    // initialise the variances of the system
    val ws = Vector.fill(ys.size)(DenseMatrix.eye[Double](dlm.f(1.0).rows))

    val theta = ffbs(dlm, ys, parameters, ws).draw
    val thetaObs = theta.map { case (t, a) => Dlm.Data(t, a.map(_.some)) }
    val fs = FactorSv.initialiseStateAr(params.factors, thetaObs, k)

    State(params, theta.toVector, fs.factors, fs.volatility)
  }

  def sample(
    priorBeta:     Gaussian,
    priorSigmaEta: InverseGamma,
    priorPhi:      Beta,
    priorMu:       Gaussian,
    priorSigma:    InverseGamma,
    priorV:        InverseGamma,
    ys:            Vector[Dlm.Data],
    dlm:           DlmModel,
    initP:         DlmFsvSystem.Parameters): Process[State] = {

    // initialise the latent state
    val init = initialise(initP, ys, dlm)

    MarkovChain(init)(sampleStep(priorBeta, priorSigmaEta, priorPhi, priorMu,
      priorSigma, priorV, ys, dlm))
  }
}
