package dlm.model

import breeze.linalg.{DenseMatrix, diag, DenseVector, inv}
import breeze.stats.distributions._
import scala.math.{exp, log}
import cats.Monad
import Dlm._

object KalmanFilter {
  case class KfState(
    time: Time,
    x: State, 
    y: Option[Observation], 
    cov: Option[DenseMatrix[Double]],
    ll: Double
  ) {
    override def toString = s"$time, ${x.mean.data.mkString(", ")}, ${x.covariance.data.mkString(", ")}, ${y.map(_.data.mkString(", ")).getOrElse("NA")}, ${cov.map(_.data.mkString(", ")).getOrElse("NA")}"
  }

  def advanceState(mod: Model, x: State, time: Time, p: Parameters): State = {
    val at = mod.g(time) * x.mean
    val rt = mod.g(time) * x.covariance * mod.g(time).t + diag(DenseVector(p.w.toArray))

    MultivariateGaussian(at, rt)
  }

  def oneStepPrediction(mod: Model, x: State, time: Time, p: Parameters) = {
    val ft = mod.f(time).t * x.mean
    val qt = mod.f(time).t * x.covariance * mod.f(time) + diag(DenseVector(p.v.toArray))

    (ft, qt)
  }

  /**
    * Update the state using Joseph Form Update given the newly observed data
    * @param 
    */
  def updateState(
    mod: Model, 
    x: State, 
    predicted: Observation, 
    y: Data, 
    p: Parameters): State = y.observation match {
    case Some(obs) =>
      val time = y.time
      val v = diag(DenseVector(p.v.toArray))
      val residual = obs - predicted
      val kalman_gain = x.covariance * mod.f(time) * inv(x.covariance)
      val mt1 = x.mean + kalman_gain * residual
      val n = p.w.size

      val identity = DenseMatrix.eye[Double](n)

      val diff = (identity - kalman_gain * mod.f(time).t)
      val covariance = diff * x.covariance * diff.t + kalman_gain * v * kalman_gain.t

      MultivariateGaussian(mt1, covariance)
    case None =>
      x
  }

  def conditionalLikelihood(
    ft: Observation, 
    qt: DenseMatrix[Double], 
    data: Option[Observation]) = data match {
    case Some(y) =>
      MultivariateGaussian(ft, qt).logPdf(y)
    case None => 0.0
  }

  def stepKalmanFilter(
    mod: Model, p: Parameters)(state: KfState, y: Data): KfState = {

    val state_prior = advanceState(mod, state.x, y.time, p)
    val (ft, qt) = oneStepPrediction(mod, state_prior, y.time, p)
    val state_posterior = updateState(mod, state_prior, ft, y, p)

    val ll = state.ll + conditionalLikelihood(ft, qt, y.observation)

    KfState(y.time, state_posterior, Some(ft), Some(qt), ll)
  }

  /**
    * Run the Kalman Filter over an array of data
    */
  def kalmanFilter(mod: Model, observations: Array[Data], p: Parameters) = {
    val init_state = MultivariateGaussian(
      DenseVector(p.m0.toArray), 
      diag(DenseVector(p.v.toArray))
    )
    val init: KfState = KfState(
      observations.head.time,
      init_state,
      None, None, 0.0)

    observations.scanLeft(init)(stepKalmanFilter(mod, p)).drop(1)
  }

  /**
    * Calculate the marginal likelihood of a DLM using a kalman filter
    */
  def kfLogLikelihood(mod: Model, observations: Array[Data])(p: Parameters): Double = {
    val init_state = MultivariateGaussian(
      DenseVector(p.m0.toArray), 
      diag(DenseVector(p.c0.toArray))
    )
    val init: KfState = KfState(
      observations.head.time,
      init_state,
      None, None, 0.0)
    observations.foldLeft(init)(stepKalmanFilter(mod, p)).ll
  }
}
