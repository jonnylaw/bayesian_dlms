package dlm.model

import Dlm._
import Smoothing._
import KalmanFilter._
import cats.implicits._
import breeze.stats.distributions.{Rand, Gamma, MarkovChain}

object GibbsSampling extends App {
  case class GibbsState(p: Parameters, state: Array[State]) {
    override def toString = p.toString
  }

  def sampleGamma(prior: Gamma, observations: Array[Double], forecasts: Array[Double]) = {
    val n = observations.size

    val difference = observations.zip(forecasts).
      map { case (y,f) => (y - f) * (y - f) }

    val shape = prior.shape + n * 0.5
    val rate = (1 / prior.scale) + difference.sum * 0.5
    val scale = 1 / rate

    Gamma(shape, scale): Rand[Double]
  }

  /**
    * Sample the (diagonal) observation noise precision matrix from a Gamma distribution
    */
  def sampleObservationMatrix(
    priorV:       Gamma,
    mod:          Model, 
    state:        Array[SmoothingState],
    observations: Array[Data]) = {

    val forecasts = state.reverse.map(x => mod.f(x.time) * x.smoothedState.mean).init

    val res = for {
      oneStepForecasts <- forecasts.map(_.data).transpose
      obs <- observations.map(_.observation).flatten.map(_.data).tail.transpose
    } yield sampleGamma(priorV, obs, oneStepForecasts)

    res.toVector.sequence
  }

  def sampleGammaState(n: Int, prior: Gamma, state: Array[Double], prevState: Array[Double]) = {
    val difference = state.zip(prevState).
      map { case (mt, mt1) => (mt - mt1) * (mt - mt1) }

    val shape = prior.shape + n * 0.5
    val rate = (1 / prior.scale) + difference.sum * 0.5
    val scale = 1 / rate

    Gamma(shape, scale): Rand[Double]
  }

  /**
    * Sample the (diagonal) system noise precision matrix from a gamma distribution
    */
  def sampleSystemMatrix(
    priorW:       Gamma,
    mod:          Model,
    state:        Array[SmoothingState], 
    observations: Array[Data]) = {

    val n = observations.size
    val states = state.map(_.smoothedState.mean)

    val res = for {
      prevStateMean <- states.map(_.data).init.transpose
      stateMean <- states.map(_.data).tail.transpose
    } yield sampleGammaState(n, priorW, stateMean, prevStateMean)

    res.toVector.sequence
  }

  def sampleState(mod: Model, observations: Array[Data], p: Parameters) = {
    val filtered = kalmanFilter(mod, observations, p)
    backwardsSmoother(mod, p)(filtered)
  }

  /**
    * A single step of a Gibbs Sampler
    */
  def dinvGammaStep(mod: Model, priorV: Gamma, priorW: Gamma, observations: Array[Data])(gibbsState: GibbsState) = {
    val state = sampleState(mod, observations, gibbsState.p)

    for {
      prec_obs <- sampleObservationMatrix(priorV, mod, state, observations)
      prec_system <- sampleSystemMatrix(priorW, mod, state, observations)
      x = state.map(_.smoothedState)
    } yield GibbsState(
      Parameters(
        prec_obs map (pv => 1/pv), 
        prec_system map (pw => 1/pw), 
        gibbsState.p.m0, 
        gibbsState.p.c0), 
      x
    )
  }

  /**
    * Do some gibbs samples
    */
  def gibbsSamples(mod: Model, priorV: Gamma, priorW: Gamma, initParams: Parameters, observations: Array[Data]) = {
    val initState = sampleState(mod, observations, initParams)
    val init = GibbsState(initParams, initState.map(_.smoothedState))

    MarkovChain(init)(dinvGammaStep(mod, priorV, priorW, observations))
  }
}
