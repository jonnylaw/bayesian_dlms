package dlm.model

import breeze.linalg.{DenseMatrix, diag, DenseVector, inv}
import breeze.stats.distributions._
import scala.math.{exp, log}
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
    time: Double,
    mt:   DenseVector[Double],
    ct:   DenseMatrix[Double],
    at:   DenseVector[Double],
    rt:   DenseMatrix[Double],
    y:    Option[DenseVector[Double]],
    cov:  Option[DenseMatrix[Double]],
    ll:   Double
  )

  /**
    * Advance the state mean and variance to the a-priori
    * value of the state at time t
    * @param g the system matrix, a function from a time increment to DenseMatrix
    * @param mt the a-posteriori mean of the state at time t-1
    * @param ct the a-posteriori covariance of the state at time t-1
    * @param dt the time increment
    * @param w the system noise matrix
    * @return the a-priori mean and covariance of the state at time t
    */
  def advanceState(
    g:    Double => DenseMatrix[Double],
    mt:   DenseVector[Double], 
    ct:   DenseMatrix[Double],
    dt:   Double, 
    w:    DenseMatrix[Double]) = {

    val at = g(dt) * mt
    val rt = g(dt) * ct * g(dt).t + w * dt

    (at, rt)
  }

  /**
    * Perform a one-step prediction
    */
  def oneStepPrediction(
    f:    Double => DenseMatrix[Double],
    at:   DenseVector[Double],
    rt:   DenseMatrix[Double],
    time: Double,
    v:    DenseMatrix[Double]) = {

    val ft = f(time).t * at
    val qt = f(time).t * rt * f(time) + v

    (ft, qt)
  }

  /**
    * Get the index of the non-missing data
    * @param y a vector of observations possibly containing missing data
    * @return a vector containing the indices of non-missing observations
    */
  def indexNonMissing[A](y: DenseVector[Option[A]]): Array[Int] = {
    y.data.map(_.isDefined).
      zipWithIndex.
      map { case (b, i) => if (b) Some(i) else None }.
      flatten
  }

  /**
    * Build observation matrix for potentially missing data
    */
  def missingF[A](
    f:    Double => DenseMatrix[Double],
    time: Double,
    y:    DenseVector[Option[A]]): DenseMatrix[Double] = {

    val missing = indexNonMissing(y)
    f(time)(::,missing.toVector).toDenseMatrix
  }

  /**
    * Perform a one-step prediction taking into account missing data
    * in the observations, this alters the size of the F-matrix
    * @param f the observation matrix
    * @param at the a-priori mean state at time t
    * @param rt the a-priori covariance of the state at time t
    * @param time the current time
    * @param v the observation variance
    * @param y the observation at time t
    */
  def oneStepMissing(
    f:    Double => DenseMatrix[Double],
    at:   DenseVector[Double],
    rt:   DenseMatrix[Double],
    time: Double,
    v:    DenseMatrix[Double],
    y:    DenseVector[Option[Double]]
  ) = {
    val fm = missingF(f, time, y)

    val ft = fm.t * at
    val qt = fm.t * rt * fm + v

    (ft, qt)
  }

  /**
    * Remove optional data from an observation vector
    * @param y a vector containing optional observations
    * @return a vector containing only the observations which are there
    */
  def flattenObs(y: DenseVector[Option[Double]]) = {
    DenseVector(y.data.flatten)
  }

  /**
    * Update the state using Joseph Form Update given the newly observed data
    * @param f the observation matrix
    * @param at the a priori state mean at time t
    * @param rt the a priori state variance at time t
    * @param d the actual observation at time t
    * @param v the variance of the measurement noise
    * @param ll the current log-likelihood
    * @return the posterior mean and variance of the latent state at time t
    */
  def updateState(
    f:         Double => DenseMatrix[Double],
    at:        DenseVector[Double],
    rt:        DenseMatrix[Double],
    d:         Data, 
    v:         DenseMatrix[Double],
    ll:        Double) = {
    
    val y = flattenObs(d.observation)
    // perform one step prediction
    val (ft, qt) = oneStepPrediction(f, at, rt, d.time, v)

    if (y.data.isEmpty) {
      (ft, qt, at, rt, ll)
    } else {
      val (predicted, predcov) = oneStepMissing(f, at, rt, d.time, v, d.observation)
      val time = d.time
      val residual = y - predicted
      
      val kalman_gain = (predcov.t \ (f(time).t * rt.t)).t
      val mt1 = at + kalman_gain * residual
      val n = mt1.size

      val identity = DenseMatrix.eye[Double](n)

      val diff = (identity - kalman_gain * f(time).t)
      val covariance = diff * rt * diff.t + kalman_gain * v * kalman_gain.t

      val newll = ll + conditionalLikelihood(predicted, predcov, y)

      (ft, qt, mt1, covariance, newll)
    }
  }

  /**
    * Calculate the conditional likelihood of the 
    * 
    */
  def conditionalLikelihood(
    ft: DenseVector[Double],
    qt: DenseMatrix[Double], 
    y:  DenseVector[Double]) = {

    if (y.size == 1) {
      Gaussian(ft(0), math.sqrt(qt(0,0))).logPdf(y(0))
    } else {
      MultivariateGaussian(ft, qt).logPdf(y)
    }
  }

  /**
    * Step the Kalman Filter a single Step
    */
  def step(
    mod:   Model, 
    p:     Parameters)(
    state: State, 
    y:     Data): State = {

    val dt = y.time - state.time
    val (at, rt) = advanceState(mod.g, state.mt, state.ct, dt, p.w)
    val (ft, qt, mt, ct, ll) = updateState(mod.f, at, rt, y, p.v, state.ll)

    State(y.time, mt, ct, at, rt, Some(ft), Some(qt), ll)
  }

  /**
    * Run the Kalman Filter over an array of data
    */
  def filter(
    mod:          Model, 
    observations: Array[Data], 
    p:            Parameters) = {
    val (at: DenseVector[Double], rt: DenseMatrix[Double]) = 
      advanceState(mod.g, p.m0, p.c0, 0, p.w)
    val init = State(
      observations.map(_.time).min - 1, 
      p.m0, p.c0, at, rt, None, None, 0.0)

    observations.scanLeft(init)(step(mod, p))
  }

  /**
    * Calculate the marginal likelihood of a DLM using a kalman filter
    */
  def logLikelihood(
    mod: Model, 
    observations: Array[Data])
    (p: Parameters): Double = {

    val (at, rt) = advanceState(mod.g, p.m0, p.c0, 0, p.w)
    val init = State(
      observations.map(_.time).min - 1,
      p.m0,
      p.c0,
      at,
      rt,
      None, None, 0.0)

    observations.foldLeft(init)(step(mod, p)).ll
  }
}
