package dlm.model

import Dlm._
import Smoothing._
import KalmanFilter._
import cats.implicits._
import breeze.linalg.{DenseVector, diag, DenseMatrix}
import breeze.stats.distributions.{Rand, MarkovChain}

object GibbsSampling extends App {
  case class State(
    p:     Parameters, 
    state: Array[(Time, DenseVector[Double])]
  )

  /**
    * Sample the (diagonal) observation noise covariance matrix from a Gamma distribution
    * This doesn't take into account missing observations
    */
  def sampleObservationMatrix(
    prior:        InverseGamma,
    mod:          Model, 
    state:        Array[(Time, DenseVector[Double])],
    observations: Array[Data]): Rand[DenseMatrix[Double]] = {

    val forecast = state.tail.
      map { case (time, x) => mod.f(time).t * x }

    val SSy = (observations.map(_.observation).flatten, forecast).zipped.
      map { case (y, f) => (y - f) *:* (y - f) }.
      reduce(_ + _)

    val shape = prior.shape + observations.size * 0.5
    val rate = SSy.map(ss => (1.0 / prior.scale) + ss * 0.5)

    val res = rate.map(r =>
      InverseGamma(shape, r).draw
    )

    Rand.always(diag(res))
  }

  /**
    * Sample the (diagonal) system noise covariance matrix from an inverse gamma 
    */
  def sampleSystemMatrix(
    prior:        InverseGamma,
    mod:          Model,
    state:        Array[(Time, DenseVector[Double])]): Rand[DenseMatrix[Double]] = {

    val advanceState = state.init.
      map { case (time, x) => mod.g(time) * x }

    // this bit is wrong
    val SStheta = (state.map(_._2).tail, advanceState).zipped.
      map { case (mt, at) => (mt - at) *:* (mt - at) }.
      reduce(_ + _)

    val shape = prior.shape + (state.size - 1) * 0.5
    val rate = SStheta map (s => (prior.scale) + s * 0.5)

    val res = rate.map(r =>
      InverseGamma(shape, r).draw
    )

    Rand.always(diag(res))
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
    priorV:       InverseGamma,
    priorW:       InverseGamma, 
    observations: Array[Data])(gibbsState: State) = {

    for {
      obs <- sampleObservationMatrix(priorV, mod, gibbsState.state, observations)
      state = sampleState(mod, observations, Parameters(obs, gibbsState.p.w, gibbsState.p.m0, gibbsState.p.c0))
      system <- sampleSystemMatrix(priorW, mod, state)
    } yield State(Parameters(gibbsState.p.v, system, gibbsState.p.m0, gibbsState.p.c0), state)
  }

  /**
    * Return a Markov chain using Gibbs Sampling to determine the values of the system and 
    * observation noise covariance matrices, W and V
    * @param mod the model containing the definition of the observation matrix F_t and system evolution matrix G_t
    * @param priorV the prior distribution on the observation noise matrix, V
    * @param priorW the prior distribution on the system noise matrix, W
    * @param initParams the initial parameters of the Markov Chain
    * @param observations an array of Data containing the observed time series
    * @return a Process 
    */
  def gibbsSamples(
    mod:          Model, 
    priorV:       InverseGamma, 
    priorW:       InverseGamma, 
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
    priorV:       InverseGamma,
    priorW:       InverseGamma, 
    observations: Array[Data])(gibbsState: State) = {

    for {
      obs <- sampleObservationMatrix(priorV, mod, gibbsState.state, observations)
      state = sampleState(mod, observations, Parameters(obs, gibbsState.p.w, gibbsState.p.m0, gibbsState.p.c0))
      system <- sampleSystemMatrix(priorW, mod, state)
      p = Parameters(obs, system, gibbsState.p.m0, gibbsState.p.c0)
      newP <- metropStep(mod, observations, proposal)(p)
    } yield State(newP, state)
  }

  def gibbsMetropSamples(
    proposal:     Parameters => Rand[Parameters],
    mod:          Model, 
    priorV:       InverseGamma, 
    priorW:       InverseGamma, 
    initParams:   Parameters, 
    observations: Array[Data]) = {

    val initState = sampleState(mod, observations, initParams)
    val init = State(initParams, initState)

    MarkovChain(init)(gibbsMetropStep(proposal, mod, priorV, priorW, observations))
  }
}
