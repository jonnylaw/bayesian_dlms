import breeze.linalg.{DenseMatrix, diag, DenseVector, inv}
import breeze.stats.distributions._
import scala.math.{exp, log}
import cats.Monad
import Dlm._

object KalmanFilter {
  case class KfState(
    x: State, 
    y: Option[Observation], 
    cov: Option[DenseMatrix[Double]],
    ll: Double
  )

  def log_to_covariance(log_p: Vector[Double]): DenseMatrix[Double] = {
    diag(DenseVector((log_p map exp _).toArray))
  }

  def advance_state(mod: Model, x: State, time: Time, p: Parameters): State = {
    val w = log_to_covariance(p.log_w)

    val at = mod.g(time) * x.mean
    val rt = mod.g(time) * x.covariance * mod.g(time).t + w

    MultivariateGaussian(at, rt)
  }

  def one_step_prediction(mod: Model, x: State, time: Time, p: Parameters) = {
    val v = log_to_covariance(p.log_v)

    val ft = mod.f(time).t * x.mean
    val qt = mod.f(time).t * x.covariance * mod.f(time) + v

    (ft, qt)
  }

  /**
    * Update the state using Joseph Form Update
    * @param 
    */
  def update_state(
    mod: Model, 
    x: State, 
    predicted: DenseVector[Double], 
    y: Data, 
    p: Parameters): State = {

    val time = y.time
    val v = log_to_covariance(p.log_v)
    val residual = y.observation - predicted
    val kalman_gain = x.covariance * mod.f(time) * inv(x.covariance)
    val mt1 = x.mean + kalman_gain * residual
    val n = p.log_w.size

    val identity = DenseMatrix.eye[Double](n)

    val covariance = (identity - kalman_gain * mod.f(time).t) * x.covariance * (identity - kalman_gain * mod.f(time).t).t + kalman_gain * v * kalman_gain.t

    MultivariateGaussian(mt1, covariance)
  }

  def conditional_likelihood(
    ft: Observation, 
    qt: DenseMatrix[Double], 
    y: Observation) = {

    MultivariateGaussian(ft, qt).logPdf(y)
  }

  def step_kalman_filter(
    mod: Model, p: Parameters)(state: KfState, y: Data): KfState = {

    val state_prior = advance_state(mod, state.x, y.time, p)
    val (ft, qt) = one_step_prediction(mod, state_prior, y.time, p)
    val state_posterior = update_state(mod, state_prior, ft, y, p)
    val ll = state.ll + conditional_likelihood(ft, qt, y.observation)

    KfState(state_posterior, Some(ft), Some(qt), ll)
  }

  /**
    * Run the Kalman Filter over an array of data
    */
  def kalman_filter(mod: Model, observations: Array[Data], p: Parameters) = {
    val init_state = MultivariateGaussian(
      DenseVector(p.m0.toArray), 
      log_to_covariance(p.log_c0)
    )
    val init: KfState = KfState(
      init_state,
      None, None, 0.0)
    observations.scanLeft(init)(step_kalman_filter(mod, p)) 
  }

  /**
    * Calculate the marginal likelihood of a DLM using a kalman filter
    */
  def kf_ll(mod: Model, observations: Array[Data])(p: Parameters): Double = {
    kalman_filter(mod, observations, p).last.ll
  }
}
