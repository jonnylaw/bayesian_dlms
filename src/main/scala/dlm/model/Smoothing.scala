package dlm.model

import Dlm._
import KalmanFilter._
import breeze.linalg.{inv, DenseVector, diag}
import breeze.stats.distributions.MultivariateGaussian

object Smoothing {
  case class SmoothingState(time: Time, state: State) {
    override def toString = s"$time, ${state.mean.data.mkString(", ")}, ${state.covariance.data.mkString(", ")}"
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
    val mt = kfState.statePosterior.mean
    val ct = kfState.statePosterior.covariance
    val at = kfState.statePrior.mean
    val rt = kfState.statePrior.covariance

    // inverse rt
    val invrt = inv(rt)

    // calculate the updated mean 
    val mean = mt + ct * mod.g(time).t * invrt * (state.state.mean - at)

    // calculate the updated covariance
    val covariance = ct - ct * mod.g(time).t * invrt * (rt - state.state.covariance) * invrt * mod.g(time) * ct

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
    val init = SmoothingState(kfState.last.time, kfState.last.statePosterior)
    kfState.init.reverse.scanLeft(init)(smoothStep(mod, p)).reverse
  }


  /**
    * Simulation version of the backwards smoother
    * @param mod 
    */
  def backSampleStep(
    mod: Model, 
    p: Parameters)(state: DenseVector[Double], 
      nextKfState: KfState, 
      curKfState: KfState) = {

    // extract elements from kalman state
    val time = curKfState.time
    val mt = curKfState.statePosterior.mean
    val ct = curKfState.statePosterior.covariance
    val at1 = nextKfState.statePrior.mean
    val rt1 = nextKfState.statePrior.covariance

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
    kfState: Array[KfState], 
    p: Parameters) = {

    val lastTime = kfState.last.time
    val lastState = kfState.last.statePosterior.draw

    kfState.init.reverse.scanLeft((kfState.last, lastTime, lastState))((x, kfs) => 
      backSampleStep(mod, p)(x._3, x._1,  kfs)
    ).reverse.map { case (_, time, state) => (time, state) }
  }
}
