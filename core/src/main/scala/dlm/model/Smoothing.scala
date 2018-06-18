package core.dlm.model

import Dlm._
import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.distributions.Rand

object Smoothing {
  case class SmoothingState(
    time:       Double, 
    mean:       DenseVector[Double], 
    covariance: DenseMatrix[Double], 
    at1:        DenseVector[Double], 
    rt1:        DenseMatrix[Double])

  /**
    * A single step in the backwards smoother
    * Requires that a Kalman Filter has been run on the model
    * @param mod a DLM model specification
    * @param state the state at time t + 1
    * @return 
    */
  def smoothStep(
    mod:     Model)
    (kfState: KalmanFilter.State,
    state:    SmoothingState) = {

    // extract elements from kalman state
    val time = kfState.time
    val dt = state.time - kfState.time
    val mt = kfState.mt
    val ct = kfState.ct
    val at1 = state.at1
    val rt1 = state.rt1

    val cgrinv = (rt1.t \ (mod.g(dt) * ct.t)).t

    val mean = mt + cgrinv * (state.mean - at1)
    val covariance = ct - cgrinv * (rt1 - state.covariance) * cgrinv

    SmoothingState(time, mean, covariance, kfState.at, kfState.rt)
  }

  /**
    * Learn the distribution of the latent state (the smoothing distribution)
    * p(x_{1:T} | y_{1:T}) of a fully specified DLM with observations available
    * for all time
    * @param mod a DLM model specification
    * @param kfState the output of a Kalman Filter
    * @return 
    */
  def backwardsSmoother(
    mod:      Model)
    (kfState: Vector[KalmanFilter.State]) = {

    val last = kfState.last
    val lastTime = last.time
    val init = SmoothingState(lastTime, last.mt, last.ct, last.at, last.rt)

    kfState.init.scanRight(init)(smoothStep(mod))
  }

  case class SamplingState(
    time:   Double, 
    sample: DenseVector[Double],
    at1:    DenseVector[Double], 
    rt1:    DenseMatrix[Double])

  def step(
    mod:      Model, 
    w:        DenseMatrix[Double])
    (kfState: KalmanFilter.State,
     state:   Smoothing.SamplingState) = {

    // extract elements from kalman state
    val dt = state.time - kfState.time
    val mt = kfState.mt
    val ct = kfState.ct
    val at1 = state.at1
    val rt1 = state.rt1

    // more efficient than inverting rt, equivalent to C * G.t * inv(R)
    val cgrinv = (rt1.t \ (mod.g(dt) * ct.t)).t

    // calculate the updated mean
    val mean = mt + cgrinv * (state.sample - at1)

    // calculate the updated covariance
    val n = w.cols
    val identity = DenseMatrix.eye[Double](n)
    val diff = identity - cgrinv * mod.g(dt)
    val covariance = diff * ct * diff.t + cgrinv * w * dt * cgrinv.t
    val r = (covariance + covariance.t) /:/ 2.0

    Smoothing.SamplingState(kfState.time,
      MultivariateGaussianSvd(mean, r).draw,
      kfState.at, 
      kfState.rt)
  }

  def initialise(filtered: Vector[KalmanFilter.State]) = {
    val last = filtered.last
    val lastState = MultivariateGaussianSvd(last.mt, last.ct).draw
    Smoothing.SamplingState(last.time, lastState, last.at, last.rt)
  }

  def sample(
    mod:      Model,
    filtered: Vector[KalmanFilter.State], 
    w:        DenseMatrix[Double]) = {

    def initState = initialise(filtered)
   
    filtered.init.
      scanRight(initState)(step(mod, w)).
      map(a => (a.time, a.sample))
  }

  /**
    * Forward filtering backward sampling for a DLM
    * @param mod the DLM
    * @param observations a list of observations
    * @param p parametes of the DLM
    */
  def ffbs(
    mod:          Model,
    observations: Vector[Data],
    p:            Dlm.Parameters) = {

    val filtered = KalmanFilter.filter(mod, observations, p)
    Rand.always(sample(mod, filtered, p.w))
  }
}
