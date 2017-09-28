package dlm.model

import Dlm._
import Smoothing._
import KalmanFilter._
import cats.implicits._
import breeze.linalg.{DenseVector, diag}
import breeze.stats.distributions.{Rand, Gamma, MarkovChain}

object GibbsSampling extends App {
  case class State(
    p:     Parameters, 
    state: Array[(Time, DenseVector[Double])]
  )

  def sampleGamma(
    prior:        Gamma, 
    observations: Array[Double], 
    forecasts:    Array[Double]) = {

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
    * TODO: Join forcasts and state on Time, instead of zipping 
    */
  def sampleObservationMatrix(
    priorV:       Gamma,
    mod:          Model, 
    state:        Array[(Time, DenseVector[Double])],
    observations: Array[Data]) = {

    val forecasts = state.map { case (time, x) => mod.f(time).t * x }

    val oneStepForecasts = forecasts.map(_.data).transpose
    val obs = observations.map(_.observation).flatten.map(_.data).transpose
    
    val res = oneStepForecasts.zip(obs).map { case (y, f) =>
      sampleGamma(priorV, y, f).draw
    }

    diag(DenseVector(res.map(pv => 1/pv)))
  }

  def sampleGammaState(
    n:         Int, 
    prior:     Gamma, 
    state:     Array[Double], 
    prevState: Array[Double]) = {

    val difference = state.zip(prevState).
      map { case (mt, mt1) => (mt1 - mt) * (mt1 - mt) }

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
    state:        Array[(Time, DenseVector[Double])], 
    observations: Array[Data]) = {

    val n = observations.size

    val prevState = state.
      map { case (time, x) => mod.g(time) * x }.
      map(_.data).transpose
    val stateMean = state.
      map { case (time, x) => x.data }.
      tail.transpose

    val res = (prevState.zip(stateMean)).map { case (mt, mt1) =>
         sampleGammaState(n, priorW, mt1, mt).draw
     }

    diag(DenseVector(res))
  }

  def sampleState(
    mod:          Model, 
    observations: Array[Data], 
    p:            Parameters) = {

    val filtered = kalmanFilter(mod, observations, p)
    backwardSampling(mod, filtered, p)
  }

  /**
    * A single step of a Gibbs Sampler
    */
  def dinvGammaStep(
    mod:          Model, 
    priorV:       Gamma,
    priorW:       Gamma, 
    observations: Array[Data])(gibbsState: State) = {

    val state = sampleState(mod, observations, gibbsState.p)

    val obs = sampleObservationMatrix(priorV, mod, state, observations)
    val prec_system = sampleSystemMatrix(priorW, mod, state, observations)
    
    Rand.always(
      State(
        Parameters(
          obs,
          prec_system map (pw => 1/pw),
          gibbsState.p.m0,
          gibbsState.p.c0),
        state
      )
    )
  }

  /**
    * Do some gibbs samples
    */
  def gibbsSamples(
    mod:          Model, 
    priorV:       Gamma, 
    priorW:       Gamma, 
    initParams:   Parameters, 
    observations: Array[Data]) = {

    val initState = sampleState(mod, observations, initParams)
    val init = State(initParams, initState)

    MarkovChain(init)(dinvGammaStep(mod, priorV, priorW, observations))
  }
}
