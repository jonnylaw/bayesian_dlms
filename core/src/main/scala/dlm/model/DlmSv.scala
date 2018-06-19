// package core.dlm.model

// import breeze.linalg.{DenseVector, DenseMatrix, diag, sum}
// import breeze.stats.distributions._
// import breeze.numerics.{log, exp}
// import cats.implicits._

// /**
//   * Model a heteroskedastic time series DLM by modelling the log-variance
//   * as a latent-state
//   */
// object DlmSv {
//   case class Parameters(
//     dlm: Dlm.Parameters,
//     sv:  StochasticVolatility.Parameters)

//     /**
//     * Simulate a single step in the DLM with time varying observation variance
//     * @param time the time of the next observation
//     * @param x the state of the DLM
//     * @param a latent state of the time varying log-variance
//     * @param dlm the DLM model to use for the evolution
//     * @param p the parameters of the DLM and FSV Model
//     * @param dt the time difference between successive observations
//     * @return the next simulated value 
//     */
//   def simStep(
//     time: Double,
//     x:    DenseVector[Double], 
//     a:    DenseVector[Double], 
//     dlm:  Dlm.Model,
//     p:    Parameters) = {

//     for {
//       wt <- MultivariateGaussian(
//         DenseVector.zeros[Double](p.dlm.w.cols), p.dlm.w)
//       at <- Dlm.stepState(a, DenseMatrix(p.sigma),
//         (t: Double) => DenseMatrix(p.phi), 1.0)
//       vt <- observation(at)
//       xt = dlm.g(1.0) * x + wt
//       y = dlm.f(time).t * xt + vt
//     } yield (Dlm.Data(time, y.map(_.some)), xt, at)
//   }

//   /**
//     * Simulate from a DLM with time varying variance represented by 
//     * a Stochastic Volatility Latent-State
//     * @param dlm the DLM model
//     * @param params the parameters of the DLM and stochastic volatility model
//     * @param p the dimension of the observations
//     * @return a Markov chain representing DLM with time evolving mean
//     */
//   def simulate(
//     dlm:    Dlm.Model,
//     params: Parameters,
//     p:      Int) = {

//     val initState = MultivariateGaussian(params.dlm.m0, params.dlm.c0).draw
//     val initAt = DenseVector(params.sigma.map(x => Gaussian(0.0, math.sqrt(x)).draw).toArray)
//     val init = (Dlm.Data(0.0, DenseVector[Option[Double]](None)), initState, initAt)

//     MarkovChain(init) { case (d, x, a) => simStep(d.time + 1.0, x, a, dlm, params) }
//   }

//   def sampleStates(
//     vs:     Vector[(Double, DenseVector[Double])],
//     alphas: Vector[(Double, DenseVector[Double])],
//     params: Parameters): Vector[(Double, DenseVector[Double])] = {

//     val times = vs.map(_._1)

//     val res: Vector[Vector[Double]] = for {
//       (v, a) <- vs.map(_._2.data).transpose zip alphas.map(_._2.data).transpose
//       a1 = sampleState(v.zip(times), a.zip(times), params)
//     } yield a1.map(_._2.data.head)

//     times zip res.transpose.map(x => DenseVector(x.toArray))
//   }

//   /**
//     * Sample the latent-state of the DLM
//     * @param 
//     */
//   def ffbs(
//     vs:     Vector[(Double, DenseVector[Double])],
//     ys:     Vector[Dlm.Data],
//     params: Dlm.Parameters,
//     mod:    Dlm.Model
//   ) = {

//     // create a list of parameters with the variance in them
//     val ps = vs.map { case (t, vi) => params.copy(v = diag(vi)) }

//     def kalmanStep(p: Dlm.Parameters) = KalmanFilter.step(mod, p) _

//     val (at, rt) = KalmanFilter.advanceState(mod.g, params.m0, 
//       params.c0, 0, params.w)
//     val init = KalmanFilter.initialiseState(mod, params, ys)

//     // fold over the list of variances and the observations
//     val filtered = (ps zip ys).
//       scanLeft(init){ case (s, (p, y)) => kalmanStep(p)(s, y) }

//     Rand.always(Smoothing.sample(mod, filtered, params.w))
//   }

//   /**
//     * Calculate y_t - F_t x_t
//     */
//   def takeMean(
//     dlm:   Dlm.Model,
//     theta: Vector[(Double, DenseVector[Double])],
//     ys:    Vector[Dlm.Data]) = {

//     for {
//       (d, x) <- ys zip theta.map(_._2)
//       fm = KalmanFilter.missingF(dlm.f, d.time, d.observation)
//       y = KalmanFilter.flattenObs(d.observation)
//     } yield (d.time, y - fm.t * x)
//   }

//   def initialiseVariances(p: Int, n: Int) = 
//     for {
//       t <- Vector.range(1, n)
//       x = DenseVector.rand(p, Gaussian(0.0, 1.0))
//     } yield (t.toDouble, x)


// }
