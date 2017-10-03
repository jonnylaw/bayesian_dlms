package dlm.model

import Dlm._
import Smoothing._
import KalmanFilter._
import cats.implicits._
import breeze.linalg.{DenseVector, diag}
import breeze.stats.distributions.{Rand, Gamma, MarkovChain}
import kantan.csv._
import kantan.csv.ops._
import kantan.csv.generic._

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

    // println(s"sum difference forecast ${difference.sum}")

    val shape = prior.shape + n * 0.5
    val rate = (1 / prior.scale) + difference.sum * 0.5
    val scale = 1 / rate

    // println(s"shape observation $shape")
    // println(s"rate observation $rate")

    Gamma(shape, scale): Rand[Double]
  }

  /**
    * Sample the (diagonal) observation noise covariance matrix from an inverse Gamma distribution
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

    // println("difference state")

    val difference = state.zip(prevState).
      map { case (mt, at) => 
        // println(s"current state $mt")
        // println(s"advanced previous state $at")
        // println(s"difference ${mt - at}")
        // println(s"difference squared ${(mt - at) * (mt - at)}")
        (mt - at) * (mt - at) }

    // println(s"difference state sum ${difference.sum}")

    val shape = prior.shape + n * 0.5
    val rate = (1 / prior.scale) + difference.sum * 0.5
    val scale = 1 / rate

    // println(s"shape system $shape")
    // println(s"rate system $rate")

    Gamma(shape, scale): Rand[Double]
  }

  /**
    * Sample the (diagonal) system noise covariance matrix from an inverse gamma 
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

    val res = (prevState.zip(stateMean)).map { case (at, mt) =>
      sampleGammaState(n, priorW, mt, at).draw
    }

    diag(DenseVector(res.map(pw => 1/pw)))
  }

  /**
    * Sample state
    */
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
    val system = sampleSystemMatrix(priorW, mod, state, observations)

    Rand.always(
      State(Parameters(obs, system, gibbsState.p.m0, gibbsState.p.c0), state)
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

  def metropStep(
    mod:          Model, 
    observations: Array[Data],
    proposal:     Parameters => Rand[Parameters]) = {

    MarkovChain.Kernels.metropolis(proposal)(KalmanFilter.logLikelihood(mod, observations))
  }

  def gibbsMetropStep(
    proposal:     Parameters => Rand[Parameters],
    mod:          Model, 
    priorV:       Gamma,
    priorW:       Gamma, 
    observations: Array[Data])(gibbsState: State) = {

    val state = sampleState(mod, observations, gibbsState.p)

    val obs = sampleObservationMatrix(priorV, mod, state, observations)
    val system = sampleSystemMatrix(priorW, mod, state, observations)
    val p = Parameters(obs, system, gibbsState.p.m0, gibbsState.p.c0)
    val newP = metropStep(mod, observations, proposal)(p)

    newP.map(State(_, state))
  }

  def gibbsMetropSamples(
    proposal:     Parameters => Rand[Parameters],
    mod:          Model, 
    priorV:       Gamma, 
    priorW:       Gamma, 
    initParams:   Parameters, 
    observations: Array[Data]) = {

    val initState = sampleState(mod, observations, initParams)
    val init = State(initParams, initState)

    MarkovChain(init)(gibbsMetropStep(proposal, mod, priorV, priorW, observations))
  }
}
