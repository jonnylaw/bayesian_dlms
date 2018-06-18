package core.dlm.model

import breeze.linalg.{DenseVector, DenseMatrix, diag, sum}
import breeze.stats.distributions._
import breeze.numerics.{log, exp}
import cats.implicits._
// import spire.syntax.cfor._

/**
  * Model a heteroskedastic time series by modelling the log-variance
  * as a latent-state
  */
object StochasticVolatility {
  private val pis = Array(0.0073, 0.1056, 0.00002, 0.044, 0.34, 0.2457, 0.2575)
  private val means = Array(-11.4, -5.24, -9.84, 1.51, -0.65, 0.53, -2.36)
  private val variances = Array(5.8, 2.61, 5.18, 0.17, 0.64, 0.34, 1.26)

  case class Parameters(
    dlm:   Dlm.Parameters,
    phi:   Vector[Double],
    sigma: Vector[Double])

  /**
    * The observation function for the stochastic volatility model
    */
  def observation(at: DenseVector[Double]): Rand[DenseVector[Double]] = {
    Rand.always(at map (a => Gaussian(0.0, 1).
      map(s => s * exp(a * 0.5)).draw))
  }

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
  def simStep(
    time: Double,
    x:    DenseVector[Double], 
    a:    DenseVector[Double], 
    dlm:  Dlm.Model,
    p:    Parameters) = {

    for {
      wt <- MultivariateGaussian(
        DenseVector.zeros[Double](p.dlm.w.cols), p.dlm.w)
      at <- Dlm.stepState(a, DenseMatrix(p.sigma),
        (t: Double) => DenseMatrix(p.phi), 1.0)
      vt <- observation(at)
      xt = dlm.g(1.0) * x + wt
      y = dlm.f(time).t * xt + vt
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
  def simulate(
    dlm:    Dlm.Model,
    params: Parameters,
    p:      Int) = {

    val initState = MultivariateGaussian(params.dlm.m0, params.dlm.c0).draw
    val initAt = DenseVector(params.sigma.map(Gaussian(0.0, math.sqrt(_)).draw).toArray)
    val init = (Dlm.Data(0.0, DenseVector[Option[Double]](None)), initState, initAt)

    MarkovChain(init) { case (d, x, a) => simStep(d.time + 1.0, x, a, dlm, params) }
  }

  /**
    * Sample the indices for the mixture model
    * @param ys a collection of observations
    * @param xs the latent log-volatility
    */
  def sampleKt(
    ys: Vector[(Double, Double)],
    xs: Vector[(Double, Double)]) = {

    // marginal likelihood
    def ll(j: Int, y: Double, x: Double) = {
      Gaussian(x, math.sqrt(variances(j))).
        logPdf(log(y * y) - means(j))
    }

    // calculate the log weights for a single index
    def logWeights(y: Double, x: Double) = for {
      j <- 0 until variances.size
    } yield log(pis(j)) + ll(j, y, x)

    for {
      (y, x) <- ys.map(_._2) zip xs.tail.map(_._2)
      lw = logWeights(y, x)
      max = lw.max
      weights = lw.map(w => exp(w - max))
      kt = Multinomial(DenseVector(weights.toArray)).draw
    } yield kt
  }

  def ar1Dlm(params: Parameters): (Dlm.Model, Dlm.Parameters) = {

    // the AR(1) model as a DLM
    val mod = Dlm.Model(
      f = (t: Double) => DenseMatrix(1.0),
      g = (dt: Double) => DenseMatrix(params.phi))

    val c0 = (params.sigma zip params.phi).
      map { case (s, p) => s * s / (1 - p) }

    val paramsSv = Dlm.Parameters(
      v = DenseMatrix(1.0),
      w = diag(DenseVector(params.sigma.toArray)),
      m0 = DenseVector.zeros[Double](params.sigma.size),
      c0 = diag(DenseVector(c0.toArray))
    )

    (mod, paramsSv)
  }

  /**
    * Sample the log-variance using a mixture model approximation
    */
  def sampleState(
    vs:     Vector[(Double, Double)],
    alphas: Vector[(Double, Double)],
    params: Parameters): Vector[(Double, DenseVector[Double])] = {

    val t0 = vs.head._1
    val dt0 = vs(1)._1 - vs.head._1
    val (mod, paramsSv) = ar1Dlm(params)

    // sample the T indices of the mixture
    val kt = sampleKt(vs, alphas)

    // construct a list of variances and means
    val vkt = kt.map(j => variances(j))
    val mkt = kt.map(j => means(j))

    val init = KalmanFilter.State(t0 - dt0, paramsSv.m0, paramsSv.c0,
      paramsSv.m0, paramsSv.c0, None, None)

    val yt = (vs zip mkt).
      map { case ((t, v), m) => (t, log(v * v) - m) }.
      map { case (t, x) => Dlm.Data(t, DenseVector(x.some)) }

    // create vector of parameters
    val ps = vkt map (newV => paramsSv.copy(v = DenseMatrix(newV)))

    val filtered = (ps zip yt).
      scanLeft(init){ case (s, (p, y)) => KalmanFilter.step(mod, p)(s, y) }

    Smoothing.sample(mod, filtered.toVector, paramsSv.w)
  }

  def sampleStates(
    vs:     Vector[(Double, DenseVector[Double])],
    alphas: Vector[(Double, DenseVector[Double])],
    params: Parameters): Vector[(Double, DenseVector[Double])] = {

    val times = vs.map(_._1)

    val res: Vector[Vector[Double]] = for {
      (v, a) <- vs.map(_._2.data).transpose zip alphas.map(_._2.data).transpose
      a1 = sampleState(v.zip(times), a.zip(times), params)
    } yield a1.map(_._2.data.head)

    times zip res.transpose.map(x => DenseVector(x.toArray))
  }

  /**
    * Log-Likelihood of the AR(1) process
    * @param state the current value of the state
    * @param p the current value of the parameters
    * @return the log-likelihood
    */
  def arLikelihood(
    state: Vector[(Double, DenseVector[Double])],
    p: Parameters): Double = {

    val n = state.length
    val chol = DenseVector(p.sigma.map(1.0 / _).toArray)

    val ssa = (state zip state.tail).
      map { case (at, at1) => at1._2 - (diag(DenseVector(p.phi.toArray)) * at._2) }.
      map (x =>  diag(chol) * x ).
      map (x => x dot x).
      sum

    val det = sum(log(chol))
    - n * 0.5 * log(2 * math.Pi) + det + ssa
  }

  /**
    * Sample Phi using a Beta prior and proposal distribution
    * @param tau a small tuning parameter for the beta proposal
    * @param lambda a tuning parameter for the beta proposal distribution
    * @param prior a prior distribution for the parameter phi
    * @return a Metropolis Hastings step sampling the value of Phi
    */
  def samplePhi(
    tau:    Double,
    lambda: Double,
    prior:  ContinuousDistr[Double],
    p:      Parameters,
    alpha:  Vector[(Double, DenseVector[Double])]) = {

    val proposal = (phi: Vector[Double]) => {
      phi traverse (x => new Beta(lambda * x + tau, lambda * (1 - x) + tau): Rand[Double])
    }

    val pos = (phi: Vector[Double]) => {
      phi.map(x => prior.logPdf(x)).sum + arLikelihood(alpha, p.copy(phi = phi))
    }

    MarkovChain.Kernels.metropolis(proposal)(pos)
  }

  /**
    * Sample sigma^2 from the an inverse gamma distribution
    * @param prior the prior for the variance of the noise of the latent-state
    * @return a distribution over the system variance
    */
  def sampleSigma(
    prior: InverseGamma,
    p:     Parameters,
    alpha: Vector[(Double, DenseVector[Double])]) = { 

    GibbsSampling.sampleSystemMatrix(prior, alpha, (t: Double) => DenseMatrix(p.phi))
  }

  /**
    * Sample the latent-state of the DLM
    * @param 
    */
  def ffbs(
    vs:     Vector[(Double, DenseVector[Double])],
    ys:     Vector[Dlm.Data],
    params: Dlm.Parameters,
    mod:    Dlm.Model
  ) = {

    // create a list of parameters with the variance in them
    val ps = vs.map { case (t, vi) => params.copy(v = diag(vi)) }

    def kalmanStep(p: Dlm.Parameters) = KalmanFilter.step(mod, p) _

    val (at, rt) = KalmanFilter.advanceState(mod.g, params.m0, 
      params.c0, 0, params.w)
    val init = KalmanFilter.initialiseState(mod, params, ys)

    // fold over the list of variances and the observations
    val filtered = (ps zip ys).
      scanLeft(init){ case (s, (p, y)) => kalmanStep(p)(s, y) }

    Rand.always(Smoothing.sample(mod, filtered, params.w))
  }

  def transformObservations(
    vs: Vector[(Double, DenseVector[Double])]): Vector[Dlm.Data] = {

    for {
      (time, vv) <- vs
      vt = vv map (v => log(v * v) + 1.27)
    } yield Dlm.Data(time, vt.map(_.some))
  }

  /**
    * Sample the initial state from the Gaussian approximation of the SV model
    * log(y_t^2) = a_t + log(e_t^2)
    * a_t = phi a_t-1 + eta_t
    * specifying the latent state evolution
    * @param p the parameters of the stochastic volatility model
    * @param ys the time series observations: y_t = e_t * exp(a_t / 2)
    * @return the latent-state a_t under the assumption that log(e_t^2) is Gaussian
    */
  def initialState(
    params: Parameters,
    vs:     Vector[(Double, DenseVector[Double])]) = {
   
    val variances = transformObservations(vs)
    val (mod, paramsSv) = ar1Dlm(params)

    Smoothing.ffbs(mod, variances,
      paramsSv.copy(v = DenseMatrix(math.Pi * math.Pi * 0.5)))
  }

  case class State(
    p: Parameters,
    theta: Vector[(Double, DenseVector[Double])],
    alpha: Vector[(Double, DenseVector[Double])])

  def takeMean(
    dlm:   Dlm.Model,
    theta: Vector[(Double, DenseVector[Double])],
    ys:    Vector[Dlm.Data]) = {

    for {
      (d, x) <- ys zip theta.map(_._2)
      fm = KalmanFilter.missingF(dlm.f, d.time, d.observation)
      y = KalmanFilter.flattenObs(d.observation)
    } yield (d.time, y - fm.t * y)
  }

  def initialiseVariances(p: Int, n: Int) = 
    for {
      t <- Vector.range(1, n)
      x = DenseVector.rand(p, Gaussian(0.0, 1.0))
    } yield (t.toDouble, x)

  def sample(
    priorSigma:   InverseGamma,
    priorPhi:     ContinuousDistr[Double],
    priorW:       InverseGamma, 
    params:       Parameters,
    observations: Vector[Dlm.Data],
    dlm:          Dlm.Model) = {

    // initialise the latent state
    val variances = initialiseVariances(observations.head.observation.size, observations.size)
    val alphas = initialState(params, variances).draw
    val initDlmState = ffbs(variances, observations, params.dlm, dlm).draw
    val init = State(params, initDlmState, alphas)

    def step(s: State) = {
      for {
        theta <- ffbs(s.alpha, observations, params.dlm, dlm)
        vs = takeMean(dlm, theta, observations)
        alpha = sampleStates(vs, s.alpha, params)
        newW <- GibbsSampling.sampleSystemMatrix(priorW, theta, dlm.g)
        newPhi <- samplePhi(0.05, 100, priorPhi, params, alpha)(s.p.phi)
        newSigma <- sampleSigma(priorSigma, params, alpha)
        p = Parameters(params.dlm.copy(w = newW), newPhi, newSigma.data.toVector)
      } yield State(p, theta.toVector, alpha)
    }

    MarkovChain(init)(step)
  }
}
