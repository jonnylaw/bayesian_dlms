package core.dlm.model

import breeze.linalg.{DenseVector, DenseMatrix, diag}
import breeze.stats.distributions._
// import breeze.numerics.{log, exp}
import cats.implicits._

/**
  * Model a heteroskedastic time series DLM by modelling the log-variance
  * as a latent-state
  */
object DlmSv {
  case class Parameters(
    dlm: Dlm.Parameters,
    sv:  Vector[StochasticVolatility.Parameters])

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
    a:    Vector[Double], 
    dlm:  Dlm.Model,
    p:    Parameters) = {

    for {
      wt <- MultivariateGaussian(
        DenseVector.zeros[Double](p.dlm.w.cols), p.dlm.w)
      at <- (a zip p.sv) traverse { case (ai, pi) =>
        StochasticVolatility.stepState(pi, ai, 1.0) }
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
  def simulate(
    dlm:    Dlm.Model,
    params: Parameters,
    p:      Int) = {

    val initState = MultivariateGaussian(params.dlm.m0, params.dlm.c0).draw
    val initAt = params.sv.map(x => Gaussian(0.0, math.sqrt(x.sigmaEta)).draw)
    val init = (Dlm.Data(0.0, DenseVector[Option[Double]](None)), initState, initAt)

    MarkovChain(init) { case (d, x, a) => simStep(d.time + 1.0, x, a, dlm, params) }
  }

  /**
    * Extract a single state from a vector of states
    * @param vs the combined state
    * @param i the position of the state to extract
    * @return the extracted state
    */
  def extractState(
    vs: Vector[(Double, DenseVector[Double])],
    i: Int): Vector[(Double, Double)] = {

    vs.map { case (t, x) => (t, x(i)) }
  }
  
  /**
    * Combine individual states into a multivariate state
    * @param s a vector of vectors containing tuples with (time, state)
    * @return a combined vector of times to state
    */
  def combineStates(
    s: Vector[Vector[(Double, Double)]]): Vector[(Double, DenseVector[Double])] = {

    s.transpose.map(x => (
      x.head._1,
      DenseVector(x.map(_._2).toArray),
    ))
  }

  /**
    * Extract the ith factor from a multivariate vector of factors
    */
  def extractFactors(
    fs: Vector[(Double, Option[DenseVector[Double]])],
    i:  Int): Vector[(Double, Option[Double])] = {

    fs map { case (t, fo) =>
      (t, fo map (f => f(i)))
    }
  }

  /**
    * Sample multiple independent log-volatility states representing the
    * time varying diagonal covariance matrix in a multivariate DLM
    * @param vs the value of the variances
    * @param alphas the current log-volatility
    * @param params the parameters of the DLM SV model
    */
  def sampleStates(
    vs:     Vector[(Double, DenseVector[Double])],
    alphas: Vector[(Double, DenseVector[Double])],
    params: Parameters): Vector[(Double, DenseVector[Double])] = {

    val times = vs.map(_._1)

    val res: Vector[Vector[Double]] = for {
      ((v, a), ps) <- vs.map(_._2.data.toVector.map(_.some)).transpose.
      zip(alphas.map(_._2.data.toVector).transpose).
      zip(params.sv)
      a1 = StochasticVolatility.sampleState(times zip v, times zip a, ps)
    } yield a1.draw.map(_._2)

    times zip res.transpose.map(x => DenseVector(x.toArray))
  }

  /**
    * Sample the latent-state of the DLM
    * @param vs the current value of the variances
    * @param ys 
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

  /**
    * Calculate y_t - F_t x_t
    */
  def takeMean(
    dlm:   Dlm.Model,
    theta: Vector[(Double, DenseVector[Double])],
    ys:    Vector[Dlm.Data]) = {

    for {
      (d, x) <- ys zip theta.map(_._2)
      fm = KalmanFilter.missingF(dlm.f, d.time, d.observation)
      y = KalmanFilter.flattenObs(d.observation)
    } yield (d.time, y - fm.t * x)
  }

  def initialiseVariances(p: Int, n: Int) = 
    for {
      t <- Vector.range(1, n)
      x = DenseVector.rand(p, Gaussian(0.0, 1.0))
    } yield (t.toDouble, x)

  case class State(
    alphas: Vector[(Double, DenseVector[Double])],
    thetas: Vector[(Double, DenseVector[Double])],
    params: Parameters)

  def initialiseState(
    params: Parameters,
    ys:     Vector[Dlm.Data],
    mod:    Dlm.Model): Rand[State] = {

    val vs = initialiseVariances(ys.head.observation.size, ys.size + 1)
    // extract the times and states individually
    val times = ys.map(_.time)
    val yse = ys.map(_.observation.data.toVector).transpose

    for {
      theta <- ffbs(vs, ys, params.dlm, mod)
      alphas <- (params.sv zip yse).
        traverse { case (pi, y) => StochasticVolatility.initialState(pi, (times zip y)) }
    } yield State(combineStates(alphas), theta, params)
  }

  def step(
    priorW:        InverseGamma,
    priorPhi:      Beta,
    priorSigmaEta: InverseGamma,
    ys:            Vector[Dlm.Data],
    mod:           Dlm.Model)(s: State): Rand[State] = ???

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
    priorW:        InverseGamma,
    priorPhi:      Beta,
    priorSigmaEta: InverseGamma,
    ys:            Vector[Dlm.Data],
    mod:           Dlm.Model,
    initP:         Parameters): Process[State] = {

    // initialise the latent state 
    val init = initialiseState(initP, ys, mod).draw

    MarkovChain(init)(step(priorW, priorPhi, priorSigmaEta, ys, mod))
  }
}
