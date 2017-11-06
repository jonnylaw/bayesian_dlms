package dlm.model

import breeze.linalg.{DenseMatrix, diag, DenseVector, inv}
import breeze.stats.distributions._
import scala.math.{exp, log}
import cats.Monad
import Dlm._

object KalmanFilter {
  /**
    * State for the Kalman Filter
    * @param time the current timestep
    * @param mt the posterior mean of the latent state 
    * @param ct the posterior covariance of the latent state
    * @param at the prior mean of the latent state
    * @param rt the prior covariance of the latent state
    * @param y the one step predicted observation mean, not present at the first timestep
    * @param cov the one step predicted observation covariance, not present at the first timestep
    * @param ll the current log-likelihood of all the observations up until the current time
    */
  case class State(
    time: Time,
    mt:   DenseVector[Double],
    ct:   DenseMatrix[Double],
    at:   DenseVector[Double],
    rt:   DenseMatrix[Double],
    y:    Option[Observation],
    cov:  Option[DenseMatrix[Double]],
    ll:   Double
  )

  def advanceState(
    g:    Time => DenseMatrix[Double], 
    mt:   DenseVector[Double], 
    ct:   DenseMatrix[Double],
    time: Time, 
    p:    Parameters) = {

    val at = g(time) * mt
    val rt = g(time) * ct * g(time).t + p.w

    (at, rt)
  }

  def oneStepPrediction(
    f:    Time => DenseMatrix[Double],
    at:   DenseVector[Double],
    rt:   DenseMatrix[Double],
    time: Time,
    p:    Parameters) = {

    val ft = f(time).t * at
    val qt = f(time).t * rt * f(time) + p.v

    (ft, qt)
  }

  /**
    * Update the state using Joseph Form Update given the newly observed data
    * @param f the observation matrix
    * @param at the a priori state mean at time t
    * @param rt the a priori state variance at time t
    * @param predicted the mean of the one-step forecast at time t
    * @param qt the variance of the one-step forecast at time t
    * @param y the actual observation at time t
    * @param p the parameters of a DLM
    * @return the posterior mean and variance of the latent state at time t
    */
  def updateState(
    f:         Time => DenseMatrix[Double],
    at:        DenseVector[Double],
    rt:        DenseMatrix[Double],
    predicted: Observation, 
    qt:        DenseMatrix[Double],
    y:         Data, 
    p:         Parameters) = y.observation match {
    case Some(obs) =>
      val time = y.time
      val residual = obs - predicted
    
      val kalman_gain = (qt.t \ (f(time).t * rt.t)).t
      val mt1 = at + kalman_gain * residual
      val n = p.w.cols

      val identity = DenseMatrix.eye[Double](n)

      val diff = (identity - kalman_gain * f(time).t)
      val covariance = diff * rt * diff.t + kalman_gain * p.v * kalman_gain.t

      (mt1, covariance)
    case None =>
      (at, rt)
  }

  def conditionalLikelihood(
    ft: Observation, 
    qt: DenseMatrix[Double], 
    data: Option[Observation]) = data match {
    case Some(y) if (y.size == 1) =>
      Gaussian(ft(0), math.sqrt(qt(0,0))).logPdf(y(0))
    case Some(y) =>
      MultivariateGaussian(ft, qt).logPdf(y)
    case None => 0.0
  }

  def stepKalmanFilter(
    mod: Model, p: Parameters)(state: State, y: Data): State = {

    val (at, rt) = advanceState(mod.g, state.mt, state.ct, y.time, p)
    val (ft, qt) = oneStepPrediction(mod.f, at, rt, y.time, p)
    val (mt, ct) = updateState(mod.f, at, rt, ft, qt, y, p)

    val ll = state.ll + conditionalLikelihood(ft, qt, y.observation)

    State(y.time, mt, ct, at, rt, Some(ft), Some(qt), ll)
  }

  /**
    * Run the Kalman Filter over an array of data
    */
  def kalmanFilter(
    mod:          Model, 
    observations: Array[Data], 
    p:            Parameters) = {
    val (at, rt) = advanceState(mod.g, p.m0, p.c0, 0, p)
    val init = State(
      observations.map(_.time).min - 1, 
      p.m0, p.c0, at, rt, None, None, 0.0)

    observations.scanLeft(init)(stepKalmanFilter(mod, p))
  }

  /**
    * Calculate the marginal likelihood of a DLM using a kalman filter
    */
  def logLikelihood(
    mod: Model, 
    observations: Array[Data])
    (p: Parameters): Double = {

    val (at, rt) = advanceState(mod.g, p.m0, p.c0, 0, p)
    val init = State(
      observations.head.time,
      p.m0,
      p.c0,
      at,
      rt,
      None, None, 0.0)

    observations.foldLeft(init)(stepKalmanFilter(mod, p)).ll
  }
}
