package dlm.model

import Dlm._
import breeze.linalg.{inv, DenseVector, DenseMatrix}
import breeze.stats.distributions.MultivariateGaussian

object Smoothing {
  case class SmoothingState(time: Time, mean: DenseVector[Double], covariance: DenseMatrix[Double])

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
    val at = kfState.at
    val rt = kfState.rt

    // inverse rt
    val invrt = inv(rt)

    // calculate the updated mean 
    val mean = mt + ct * mod.g(time).t * invrt * (state.mean - at)

    // calculate the updated covariance
    val covariance = ct - ct * mod.g(time).t * invrt * (rt - state.covariance) * invrt * mod.g(time) * ct

    SmoothingState(time, mean, covariance)
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
    val init = SmoothingState(kfState.last.time, kfState.last.mt, kfState.last.ct)
    kfState.init.reverse.scanLeft(init)(smoothStep(mod, p)).reverse
  }

  def backSampleStepJoseph(
    mod: Model,
    p: Parameters)(state: DenseVector[Double], 
      nextKfState: KalmanFilter.State, 
      curKfState: KalmanFilter.State) = {

    // extract elements from kalman state
    val time = curKfState.time
    val mt = curKfState.mt
    val ct = curKfState.ct
    val at1 = nextKfState.at
    val rt1 = nextKfState.rt

    val cgrinv = ct * mod.g(time + 1) * inv(rt1)

    // calculate the updated mean
    // the difference between the backwards sampler and smoother is here
    // we take the difference of the previously sampled state 
    val mean = mt + cgrinv * (state - at1)

    // calculate the updated covariance
    val n = p.w.cols
    val identity = DenseMatrix.eye[Double](n)
    val diff = (identity - cgrinv * mod.g(time + 1))
    val covariance = diff * ct * diff.t + cgrinv * p.w * cgrinv.t

    (curKfState, curKfState.time, MultivariateGaussian(mean, covariance).draw)
  }

  /**
    * Simulation version of the backwards smoother
    * @param mod 
    */
  def backSampleStep(
    mod: Model, 
    p: Parameters)(state: DenseVector[Double], 
      nextKfState: KalmanFilter.State, 
      curKfState: KalmanFilter.State) = {

    // extract elements from kalman state
    val time = curKfState.time
    val mt = curKfState.mt
    val ct = curKfState.ct
    val at1 = nextKfState.at
    val rt1 = nextKfState.rt

    // inverse rt
    val invrt = inv(rt1)

    // calculate the updated mean
    // the difference between the backwards sampler and smoother is here
    // we take the difference of the previously sampled state 
    val mean = mt + ct * mod.g(time + 1).t * invrt * (state - at1)

    // calculate the updated covariance
    val covariance = ct - ct * mod.g(time + 1).t * invrt * mod.g(time + 1) * ct

    (curKfState, curKfState.time, MultivariateGaussian(mean, covariance).draw)
  }

  def backwardSampling(
    mod: Model,
    kfState: Array[KalmanFilter.State], 
    p: Parameters) = {

    val lastTime = kfState.last.time
    val lastState = MultivariateGaussian(kfState.last.mt, kfState.last.ct).draw

    kfState.init.reverse.scanLeft((kfState.last, lastTime, lastState))((x, kfs) => 
      backSampleStepJoseph(mod, p)(x._3, x._1,  kfs)
    ).reverse.map { case (_, time, state) => (time, state) }
  }
}
