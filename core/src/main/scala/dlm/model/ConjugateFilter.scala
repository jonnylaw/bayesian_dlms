package core.dlm.model

import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions._
import cats.Traverse
import cats.implicits._

/**
  * State for the conjugate filter
  * @param time the time of the observation
  * @param mt the posterior mean of the latent state 
  * @param ct the posterior covariance of the latent state
  * @param at the prior mean of the latent state
  * @param rt the prior covariance of the latent state
  * @param ft the one-step forecast mean
  * @param qt the one-step forecast covariance
  * @param precision the distribution of the observation precision 
  * @param ll the running marginal log-likelihood
  */
case class GammaState(
  time:      Double,
  mt:        DenseVector[Double],
  ct:        DenseMatrix[Double],
  at:        DenseVector[Double],
  rt:        DenseMatrix[Double],
  ft:        Option[DenseVector[Double]],
  qt:        Option[DenseMatrix[Double]],
  variance:  Vector[InverseGamma],
  ll:        Double)

/**
  * Calculate an one-dimensional unknown observation variance 
  */
case class ConjugateFilter(prior: InverseGamma) extends Filter[GammaState, DlmParameters, DlmModel] {

  def initialiseState[T[_]: Traverse](
    model: DlmModel,
    p: DlmParameters,
    ys: T[Dlm.Data]): GammaState = {

    val t0 = ys.map(_.time).reduceLeftOption((t0, d) => math.min(t0, d))
    val p0 = Vector.fill(p.v.cols)(prior)
    GammaState(t0.get - 1.0, p.m0, p.c0, p.m0, p.c0, None, None, p0, 0.0)
  }

  def diagonal(m: DenseMatrix[Double]): Vector[Double] = {
    for {
      i <- Vector.range(0, m.cols)
    } yield m(i, i)
  }

  def updateStats(
    prior: Vector[InverseGamma],
    qt:    DenseMatrix[Double],
    e:     DenseVector[Double],
    v:     DenseMatrix[Double]): Vector[InverseGamma] = {

    val shapes = prior map (_.shape + 1)
    val scales = diag(DenseVector(prior.map(_.scale).toArray)) + qt.t \ (v * (e * e.t))

    (diagonal(scales) zip shapes) map { case (sc, sh) => InverseGamma(sh, sc) }
  }

  def meanVariance(vs: Vector[InverseGamma]): DenseMatrix[Double] = {
    diag(DenseVector(vs.map(_.mean).toArray))
  }

  def step(
    mod: DlmModel,
    p: DlmParameters,
    advState: (GammaState, Double) => GammaState)
    (s: GammaState, d: Dlm.Data): GammaState = {

    // calculate the time difference
    val dt = d.time - s.time
    val f = mod.f(d.time)

    // calculate moments of the advanced state, prior to the observation
    val st = advState(s, dt)

    // calculate the mean of the forecast distribution for y
    val v = meanVariance(s.variance)
    val ft = f.t * st.at
    val qt = f.t * st.rt * f + v

    // calculate the difference between the mean of the
    // predictive distribution and the actual observation
    val y = KalmanFilter.flattenObs(d.observation)
    val e = y - ft

    // calculate the kalman gain 
    val k = (qt.t \ (f.t * st.rt.t)).t
    val mt1 = st.at + k * e
    val n = mt1.size

    val vs = updateStats(s.variance, qt, e, v)
    // compute joseph form update to covariance
    val i = DenseMatrix.eye[Double](n)
    val covariance = (i - k * f.t) * st.rt * (i - k * f.t).t + k * meanVariance(vs) * k.t

    // update the marginal likelihood
    val newll = s.ll + KalmanFilter.conditionalLikelihood(ft, qt, y)
    val m = st.mt + k * e

    // In Breeze, Gamma is parameterised using shape and scale (scale = 1/rate)
    GammaState(d.time, m, covariance, st.at, st.rt,  Some(ft), Some(qt), vs, newll)
  }
}

object ConjugateFilter {
  /**
    * Advance the state mean and variance to the a-priori
    * value of the state at time t
    * @param g the system matrix, a function from a time increment to DenseMatrix
    * @param s the current state of the Kalman Filter
    * @param dt the time increment
    * @return the a-priori mean and covariance of the state at time t
    */
  def advanceState(
    p: DlmParameters,
    g: Double => DenseMatrix[Double])
    (s:  GammaState,
     dt: Double) = {

    val (at, rt) = KalmanFilter.advState(g, s.mt, s.ct, dt, p.w)
    s.copy(at = at, rt = rt)
  }
}


