package dlm.core.model

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions._
import cats.Traverse
import cats.implicits._
import math.exp

/**
  * State for the Kalman Filter
  * @param time the current timestep
  * @param mt the posterior mean of the latent state
  * @param ct the posterior covariance of the latent state
  * @param at the prior mean of the latent state
  * @param rt the prior covariance of the latent state
  * @param dt the one step predicted observation mean
  * not present at the first timestep
  * @param qt the one step predicted observation covariance,
  * not present at the first timestep
  */
case class KfState(
  time: Double,
  mt: DenseVector[Double],
  ct: DenseMatrix[Double],
  at: DenseVector[Double],
  rt: DenseMatrix[Double],
  ft: Option[DenseVector[Double]],
  qt: Option[DenseMatrix[Double]],
  ll: Double
)

case class KalmanFilter(advState: (KfState, Double) => KfState)
    extends FilterTs[KfState, DlmParameters, Dlm] {
  import KalmanFilter._

  /**
    * Perform a one-step prediction taking into account missing data
    * in the observations, this alters the size of the F-matrix
    * @param fm the observation matrix encoded for missing ness
    * @param vm the observation variance encoded for missingness
    * @param at the a-priori mean state at time t
    * @param rt the a-priori covariance of the state at time t
    */
  def oneStepMissing(
    fm: DenseMatrix[Double],
    vm: DenseMatrix[Double],
    at: DenseVector[Double],
    rt: DenseMatrix[Double]) = {

    val ft = fm.t * at
    val qt = fm.t * rt * fm + vm

    (ft, qt)
  }

  /**
    * Update the state using Joseph Form Update given the newly observed data
    * @param f the observation matrix
    * @param at the a priori state mean at time t
    * @param rt the a priori state variance at time t
    * @param d the actual observation at time t
    * @param v the variance of the measurement noise
    * @return the posterior mean and variance of the latent state at time t
    */
  def updateState(
    f: Double => DenseMatrix[Double],
    at: DenseVector[Double],
    rt: DenseMatrix[Double],
    d: Data,
    v: DenseMatrix[Double],
    ll: Double) = {

    val y = flattenObs(d.observation)
    // perform one step prediction
    val (ft, qt) = oneStepPrediction(f, at, rt, d.time, v)

    if (y.data.isEmpty) {
      (ft, qt, at, rt, ll)
    } else {
      val vm = missingV(v, d.observation)
      val fm = missingF(f, d.time, d.observation)
      val (predicted, predcov) = oneStepMissing(fm, vm, at, rt)

      val residual = y - predicted

      val kalmanGain = (predcov.t \ (fm.t * rt.t)).t
      val mt1 = at + kalmanGain * residual
      val n = mt1.size

      val identity = DenseMatrix.eye[Double](n)

      val diff = (identity - kalmanGain * fm.t)
      val covariance = diff * rt * diff.t + kalmanGain * vm * kalmanGain.t
      val newll = ll + conditionalLikelihood(predicted, predcov, y)

      (ft, qt, mt1, covariance, newll)
    }
  }

  /**
    * Step the Kalman Filter a single Step
    */
  def step(
    mod:    Dlm,
    p:      DlmParameters)
    (state: KfState,
     y:     Data): KfState = {

    val dt = y.time - state.time
    val st = advState(state, dt)
    val (ft, qt, mt, ct, ll) =
      updateState(mod.f, st.at, st.rt, y, p.v, state.ll)

    KfState(y.time, mt, ct, st.at, st.rt, Some(ft), Some(qt), ll)
  }

  /**
    * Initialise the state of the Kalman Filter
    */
  def initialiseState[T[_]: Traverse](
    mod: Dlm,
    p: DlmParameters,
    obs: T[Data]): KfState = {

    val t0 = obs.map(_.time).reduceLeftOption((t0, d) => math.min(t0, d))
    KfState(t0.get - 1.0, p.m0, p.c0, p.m0, p.c0, None, None, 0.0)
  }

  def transformParams(p: DlmParameters): DlmParameters = p
}

object KalmanFilter {
  /**
    * Remove optional data from an observation vector
    * @param y a vector containing optional observations
    * @return a vector containing only the observations which are there
    */
  def flattenObs(y: DenseVector[Option[Double]]) = {
    DenseVector(y.data.flatten)
  }

  /**
    * Calculate the conditional likelihood of the observation given the
    * corresponding one-step forecast
    */
  def conditionalLikelihood(ft: DenseVector[Double],
                            qt: DenseMatrix[Double],
                            y: DenseVector[Double]) = {

    if (y.size == 1) {
      Gaussian(ft(0), math.sqrt(qt(0, 0))).logPdf(y(0))
    } else {
      MultivariateGaussian(ft, qt).logPdf(y)
    }
  }

  /**
    * Get the index of the non-missing data
    * @param y a vector of observations possibly containing missing data
    * @return a vector containing the indices of non-missing observations
    */
  def indexNonMissing[A](y: DenseVector[Option[A]]): Vector[Int] = {
    y.data
      .map(_.isDefined)
      .zipWithIndex
      .map { case (b, i) => if (b) Some(i) else None }
      .toVector
      .flatten
  }

  /**
    * Build observation matrix for potentially missing data
    */
  def missingF[A](f: Double => DenseMatrix[Double],
                  time: Double,
                  y: DenseVector[Option[A]]): DenseMatrix[Double] = {

    val missing = indexNonMissing(y)
    f(time)(::, missing.toVector).toDenseMatrix
  }

  /**
    * Build observation error variance matrix for potentially missing data
    */
  def missingV[A](v: DenseMatrix[Double],
                  y: DenseVector[Option[A]]): DenseMatrix[Double] = {

    val missing = indexNonMissing(y)
    v(missing.toVector, missing.toVector).toDenseMatrix
  }

  /**
    * Filter the first order autoregressive model
    * Y_t = a_t + v_t, v_t ~ N(0, v)
    * a_t = mu + phi * (a_t - mu) + eta_t, eta_t ~ N(0, sigmaEta^2)
    * @param ys
    */
  def filterAr[T[_]: Traverse](
    ys: T[Data],
    p:  SvParameters,
    v:  Double) = {

    val mod = Dlm.polynomial(1)
    val dlmP = DlmParameters(
      DenseMatrix(v),
      DenseMatrix.eye[Double](1),
      DenseVector.zeros[Double](1),
      DenseMatrix.eye[Double](1))

    KalmanFilter(advanceStateAr(p)).filterTraverse(mod, ys, dlmP)
  }

  /**
    * Filter the ornstein-uhlenbeck process
    * Y_t = a_t + v_t, v_t ~ N(0, v)
    * da_t = phi * (a_t - mu) dt + sigma_eta dW_t
    * @param ys
    */
  def filterOu[T[_]: Traverse](
    ys: T[Data],
    p:  SvParameters,
    v:  Double) = {

    val mod = Dlm.polynomial(1)
    val dlmP = DlmParameters(
      DenseMatrix(v),
      DenseMatrix.eye[Double](1),
      DenseVector.zeros[Double](1),
      DenseMatrix.eye[Double](1))

    KalmanFilter(advanceStateOu(p)).filterTraverse(mod, ys, dlmP)
  }

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
    (s: KfState,
     dt: Double): KfState = {

    val (at, rt) = advState(g, s.mt, s.ct, dt, p.w)
    s.copy(at = at, rt = rt)
  }

  /**
    * Advance the state 
    */
  def advState(
    g: Double => DenseMatrix[Double],
    mt: DenseVector[Double],
    ct: DenseMatrix[Double],
    dt: Double,
    w:  DenseMatrix[Double]) = {

    if (dt == 0) {
      (mt, ct)
    } else {
      val at = g(dt) * mt
      val rt = g(dt) * ct * g(dt).t + w * dt
      (at, rt)
    }
  }

  /**
    * Filter a DLM with a random walk latent-state
    */
  def filterDlm[T[_]: Traverse](
    mod: Dlm,
    ys:  T[Data],
    p:   DlmParameters) = {

    KalmanFilter(advanceState(p, mod.g)).filterTraverse(mod, ys, p)
  }

  /**
    * Calculate the marginal likelihood of a DLM using a kalman filter
    */
  def likelihood[T[_]: Traverse](
    mod:      Dlm,
    ys:       T[Data])(
    p:        DlmParameters): Double = {

    val kf = KalmanFilter(advanceState(p, mod.g))
    val init = kf.initialiseState(mod, p, ys)

    ys.foldLeft(init)(kf.step(mod, p)).ll
  }

  /**
    * Perform a one-step prediction
    */
  def oneStepPrediction(
    f: Double => DenseMatrix[Double],
    at: DenseVector[Double],
    rt: DenseMatrix[Double],
    time: Double,
    v: DenseMatrix[Double]) = {

    val ft = f(time).t * at
    val qt = f(time).t * rt * f(time) + v

    (ft, qt)
  }

/**
  * Single step of a univariate kalman filter
  */
  def stepUni(
    d:   (Double, Option[Double]),
    t0:  Double,
    mt:  Double,
    ct:  Double,
    p:   DlmParameters,
    mod: Dlm) = {

  val t = d._1
  // extract model "matrices"
  val dt = t - t0
  val g = mod.g(dt)(0,0)
  val f = mod.f(t)(0,0)

  val at = g * mt
  val rt = g * g * ct + p.w(0,0) * dt
  val ft = f * at
  val qt = f * f * rt + p.v(0,0)

  d._2 match {
    case Some(y) =>
      val kt = f * rt / qt
      val et = y - ft

      (t, at + kt * et, kt * p.v(0,0))
    case None => (t, at, rt)
  }
}

  /**
    * Univariate Kalman Filter for the first-order polynomial model
    */
  def univariateKf(
    ys: Vector[(Double, Option[Double])],
    p:  DlmParameters,
    mod: Dlm) = {

    ys.scanLeft((ys.head._1, p.m0(0), p.c0(0,0))){ case ((t, mt, ct), y) =>
      stepUni(y, t, mt, ct, p, mod) }
  }

  /**
    * Advance the state mean and variance to the a-priori
    * value of the state at time t
    * @param st the kalman filter state
    * @param dt the time difference between observations (1.0)
    * @param p the parameters of the SV Model
    * @return the a-priori mean and covariance of the state at time t
    */
  def advanceStateAr(
    p:  SvParameters)(
    st: KfState,
    dt: Double) = st.copy(
      time = st.time + dt,
      at = st.mt map (m => p.mu + p.phi * (m - p.mu)),
      rt = st.ct map (c => p.phi * p.phi * c + p.sigmaEta * p.sigmaEta))

  /**
    * Advance the distribution of the state using the OU process
    * @param svp the parameters of the stochastic volatility model
    * @param st the state at time t
    * @param dt the time difference to the next observation or point of interest
    * @param p the parameters of the DlM
    * @return the a-priori distribution of the latent state at time t + dt 
    */
  def advanceStateOu(
    p: SvParameters)(
    st: KfState,
    dt: Double): KfState = {

    val variance = (p.sigmaEta / (2 * p.phi)) * (1 - exp(-2 * p.phi * dt))
    val identity = DenseMatrix.eye[Double](st.mt.size)

    st.copy(time = st.time + dt,
      at = st.mt map (m => p.mu + exp(p.phi * dt) * (m - p.mu)),
      rt = (identity * exp(-p.phi * dt)) * st.ct + (identity * variance))
  }
}
