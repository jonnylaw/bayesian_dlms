package dlm.model

import Dlm._
import KalmanFilter._
import breeze.linalg.inv
import breeze.stats.distributions.MultivariateGaussian

object Smoothing {
  case class SmoothingState(time: Time, smoothedState: State) {
    override def toString = s"$time, ${smoothedState.mean.data.mkString(", ")}, ${smoothedState.covariance.data.mkString(", ")}"
  }

  /**
    * A single step in the backwards smoother
    * Requires that a Kalman Filter has been run on the model
    * @param mod a DLM model specification
    * @param state the state at time t + 1
    * @param p the parameters of the DLM
    * @return 
    */
  def smoothStep(mod: Model, p: Parameters)(state: SmoothingState, kfState: KfState) = {
    // extract elements from kalman state
    val time = kfState.time
    val mt = kfState.x.mean
    val ct = kfState.x.covariance

    // advance the state and extract components
    val MultivariateGaussian(at, rt) = advanceState(mod, kfState.x, time, p)

    // inverse rt
    val invrt = inv(rt)

    // calculate the updated mean 
    val mean = mt + ct * mod.g(time).t * invrt * (state.smoothedState.mean - at)

    // calculate the updated covariance
    val covariance = ct - ct * mod.g(time).t * invrt * (rt - state.smoothedState.covariance) * invrt * mod.g(time) * ct

    SmoothingState(time, MultivariateGaussian(mean, covariance))
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
  def backwardsSmoother(mod: Model, p: Parameters)(kfState: Array[KfState]) = {
    val init = SmoothingState(kfState.last.time, kfState.last.x)
    kfState.init.reverse.scanLeft(init)(smoothStep(mod, p))
  }
}
