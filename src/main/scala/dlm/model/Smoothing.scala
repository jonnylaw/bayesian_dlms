package dlm.model

import Dlm._
import breeze.linalg.{inv, DenseVector, DenseMatrix}
import breeze.stats.distributions.MultivariateGaussian

object Smoothing {
  case class SmoothingState(
    time:       Time, 
    mean:       DenseVector[Double], 
    covariance: DenseMatrix[Double], 
    at1:        DenseVector[Double], 
    rt1:        DenseMatrix[Double])

  /**
    * A single step in the backwards smoother
    * Requires that a Kalman Filter has been run on the model
    * @param mod a DLM model specification
    * @param state the state at time t + 1
    * @param p the parameters of the DLM
    * @return 
    */
  def smoothStep(mod: Model, p: Parameters)(state: SmoothingState, kfState: KalmanFilter.State) = {
    // extract elements from kalman state
    val time = kfState.time
    val mt = kfState.mt
    val ct = kfState.ct
    val at = state.at1
    val rt = state.rt1

    val invrt = inv(rt)

    // calculate the updated mean 
    val mean = mt + ct * mod.g(time).t * invrt * (state.mean - at)

    // calculate the updated covariance
    val covariance = ct - ct * mod.g(time).t * invrt * (rt - state.covariance) * invrt * mod.g(time) * ct

    SmoothingState(time, mean, covariance, kfState.at, kfState.rt)
  }

  /**
    * Learn the distribution of the latent state (the smoothing distribution)
    * p(x_{1:T} | y_{1:T}) of a fully specified DLM with observations available
    * for all time
    * @param mod a DLM model specification
    * @param p the parameters of the DLM
    * @param kfState the output of a Kalman Filter
    * @return
    */
  def backwardsSmoother(mod: Model, p: Parameters)(kfState: Array[KalmanFilter.State]) = {

    val sortedState = kfState.sortWith(_.time > _.time)
    val last = sortedState.head
    val lastTime = last.time

    sortedState.tail.scanLeft(SmoothingState(lastTime, last.mt, last.ct, last.at, last.rt))(smoothStep(mod, p)).
      sortBy(_.time)
  }

  case class SamplingState(
    time:       Time, 
    sample:     DenseVector[Double],
    at1:        DenseVector[Double], 
    rt1:        DenseMatrix[Double])

  /**
    * Simulation version of the backwards smoother
    * @param mod 
    */
  def backSampleStep(
    mod: Model, 
    p: Parameters)(state: SamplingState, kfState: KalmanFilter.State) = {

    // extract elements from kalman state
    val time = kfState.time
    val mt = kfState.mt
    val ct = kfState.ct
    val at1 = state.at1
    val rt1 = state.rt1

    val invrt = inv(rt1)

    // calculate the updated mean
    // the difference between the backwards sampler and smoother is here
    // we take the difference of the previously sampled state 
    val mean = mt + ct * mod.g(time + 1).t * invrt * (state.sample - at1)

    // calculate the updated covariance
    val covariance = ct - ct * mod.g(time + 1).t * invrt * mod.g(time + 1) * ct

    SamplingState(kfState.time, MultivariateGaussianSvd(mean, covariance).draw, kfState.at, kfState.rt)
  }

  /**
    * Backwards sample step from the distribution p(x_t | x_{t-1}, y_{1:t}), for use in the Gibbs Sampler
    * This uses the joseph form of the covariance update for stability 
    */
  def backSampleStepJoseph(
    mod: Model,
    p: Parameters)(state: SamplingState, kfState: KalmanFilter.State) = {

    // extract elements from kalman state
    val time = kfState.time
    val mt = kfState.mt
    val ct = kfState.ct
    val at1 = state.at1
    val rt1 = state.rt1

    // more efficient than inverting rt, equivalent to C * G.t * inv(R)
    val cgrinv = (rt1 \ (mod.g(time + 1) * ct)).t

    // calculate the updated mean
    // the difference between the backwards sampler and smoother is here
    // we take the difference of the previously sampled state and the next states
    // advanced state mean
    val mean = mt + cgrinv * (state.sample - at1)

    // calculate the updated covariance
    val n = p.w.cols
    val identity = DenseMatrix.eye[Double](n)
    val diff = identity - cgrinv * mod.g(time + 1)
    val covariance = diff * ct * diff.t + cgrinv * p.w * cgrinv.t

    // calculate the updated covariance
    // val covariance = ct - cgrinv * mod.g(time + 1) * ct

    SamplingState(kfState.time, MultivariateGaussianSvd(mean, covariance).draw, kfState.at, kfState.rt)
  }

  def backwardSampling(
    mod: Model,
    kfState: Array[KalmanFilter.State], 
    p: Parameters) = {

    // sort the state in reverse order
    val sortedState = kfState.sortWith(_.time > _.time)

    // extract the final state
    val last = sortedState.head
    val lastTime = last.time
    val lastState = MultivariateGaussianSvd(last.mt, last.ct).draw

    sortedState.tail.scanLeft(SamplingState(lastTime, lastState, last.at, last.rt))(backSampleStepJoseph(mod, p)).
      sortBy(_.time).map(a => (a.time, a.sample))
  }
}
