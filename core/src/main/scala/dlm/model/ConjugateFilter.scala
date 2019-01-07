package com.github.jonnylaw.dlm

import breeze.linalg.{DenseMatrix, DenseVector, diag}
import cats.Traverse
import cats.implicits._

/**
  * State for the conjugate filter
  * @param kfState the latent-state
  * @param variance the distribution of the observation precision
  */
case class InverseGammaState(kfState: KfState, variance: Vector[InverseGamma])

/**
  * Calculate an one-dimensional unknown observation variance
  */
case class ConjugateFilter(
    prior: InverseGamma,
    advState: (InverseGammaState, Double) => InverseGammaState)
    extends FilterTs[InverseGammaState, DlmParameters, Dlm] {

  def initialiseState[T[_]: Traverse](model: Dlm,
                                      p: DlmParameters,
                                      ys: T[Data]): InverseGammaState = {

    val t0 = ys.map(_.time).reduceLeftOption((t0, d) => math.min(t0, d))
    val p0 = Vector.fill(p.v.cols)(prior)
    InverseGammaState(KfState(t0.get - 1.0, p.m0, p.c0, p.m0, p.c0, None, None),
                      p0)
  }

  def diagonal(m: DenseMatrix[Double]): Vector[Double] = {
    for {
      i <- Vector.range(0, m.cols)
    } yield m(i, i)
  }

  def updateStats(prior: Vector[InverseGamma],
                  qt: DenseMatrix[Double],
                  e: DenseVector[Double],
                  v: DenseMatrix[Double]): Vector[InverseGamma] = {

    val shapes = prior map (_.shape + 1)
    val scales = diag(DenseVector(prior.map(_.scale).toArray)) + qt.t \ (v * (e * e.t))

    (diagonal(scales) zip shapes) map { case (sc, sh) => InverseGamma(sh, sc) }
  }

  def meanVariance(vs: Vector[InverseGamma]): DenseMatrix[Double] = {
    diag(DenseVector(vs.map(_.mean).toArray))
  }

  def step(mod: Dlm, p: DlmParameters)(s: InverseGammaState,
                                       d: Data): InverseGammaState = {

    // calculate the time difference
    val dt = d.time - s.kfState.time
    val f = mod.f(d.time)

    // calculate moments of the advanced state, prior to the observation
    val st = advState(s, dt)

    // calculate the mean of the forecast distribution for y
    val v = meanVariance(s.variance)
    val ft = f.t * st.kfState.at
    val qt = f.t * st.kfState.rt * f + v

    // calculate the difference between the mean of the
    // predictive distribution and the actual observation
    val y = KalmanFilter.flattenObs(d.observation)
    val e = y - ft

    // calculate the kalman gain
    val k = (qt.t \ (f.t * st.kfState.rt.t)).t
    val mt1 = st.kfState.at + k * e
    val n = mt1.size

    val vs = updateStats(s.variance, qt, e, v)
    // compute joseph form update to covariance
    val i = DenseMatrix.eye[Double](n)
    val covariance = (i - k * f.t) * st.kfState.rt * (i - k * f.t).t + k * v * k.t

    // update the marginal likelihood
    val m = st.kfState.mt + k * e

    InverseGammaState(KfState(d.time,
                              m,
                              covariance,
                              st.kfState.at,
                              st.kfState.rt,
                              Some(ft),
                              Some(qt)),
                      vs)
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
  def advanceState(p: DlmParameters, g: Double => DenseMatrix[Double])(
      s: InverseGammaState,
      dt: Double) = {

    s.copy(kfState = KalmanFilter.advanceState(p, g)(s.kfState, dt))
  }
}
