package core.dlm.model

import breeze.linalg.{DenseVector, DenseMatrix, diag}
import breeze.stats.distributions._
import breeze.stats.mean
// import breeze.numerics.{log, exp}
import cats.implicits._

/**
  * Fit a DLM with the system variance modelled using an FSV model
  * and latent log volatility modelled using continuous time Ornstein-Uhlenbeck process
  */
object DlmFsvSystem {
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
    factors: FactorSv.Parameters)

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
    dimObs: Int
  ) = {
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
    dimObs: Int
  ) = {
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
    * Calculate the variance of the factor volatility model
    * beta f_t + eps, eps ~ N(0, sigma_x)
    * @param beta the factor loading matrix
    * @param sigmaX the small error not accounted
    * for in the factor structure
    * @param ps the log-volatility
    */
  def factorVariance(
    beta:   DenseMatrix[Double],
    sigmaX: Double,
    ps:     Vector[SvParameters]
  ) = {

    val arVar = for {
      (mu, sigmaEta, phi) <- ps.map(p => (p.mu, p.sigmaEta, p.phi))
    } yield math.exp(mu + 0.5 * sigmaEta / (1 - phi * phi))

    beta * diag(DenseVector(arVar.toArray)) * beta.t +
      diag(DenseVector.fill(beta.rows)(sigmaX))
  }

  /**
    * Center the observations to taking away the dynamic mean of the series
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

  def toDlmParameters(
    p:      DlmFsvSystem.Parameters,
    dimObs: Int): DlmParameters = {

    DlmParameters(
      diag(DenseVector.fill(dimObs)(p.v)),
      factorVariance(p.factors.beta, p.factors.v, p.factors.factorParams),
      p.m0,
      p.c0)
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
    * Perform a single step of the Gibbs Sampling algorithm
    * for the DLM FSV where the system and observation variance is modelled
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
      dlmP = toDlmParameters(s.p, p)
      theta <- SvdSampler.ffbsDlm(dlm, ys, dlmP)
      newV <- Rand.always(1.0) // sampleObservationVariance(priorV, dlm.f, theta, ys)
      newP = s.p.copy(factors = fs1.params, v = newV)
    } yield State(newP, theta, fs1.factors, fs1.volatility)
  }

  def initialise(
    params: DlmFsvSystem.Parameters,
    ys:     Vector[Dlm.Data],
    dlm:    DlmModel
  ) = {

    val k = params.factors.beta.cols
    val p = ys.head.observation.size

    val theta = SvdSampler.ffbsDlm(dlm, ys, toDlmParameters(params, p)).draw
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
