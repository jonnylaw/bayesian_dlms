package dlm.model

import breeze.stats.distributions._
import breeze.linalg._
import Dlm._

/**
  * This class learns a correlated system matrix using the Wishart prior on the precision matrix
  */
object GibbsWishart {
  /**
    * Sample a diagonal observation covariance matrix from the d-inverse Gamma Prior
    */
  def sampleObservationMatrix(
    priorV:       Gamma, 
    mod:          Model, 
    state:        Array[(Time, DenseVector[Double])], 
    observations: Array[Data]) = {

    GibbsSampling.sampleObservationMatrix(priorV, mod, state, observations).map(1/_)
  }

  /**
    * Sample the system covariance matrix using a Wishart prior on the system precision matrix
    */
  def sampleSystemMatrix(
    priorW:       Wishart, 
    mod:          Model, 
    state:        Array[(Time, DenseVector[Double])], 
    observations: Array[Data]): Rand[DenseMatrix[Double]] = {

    val n = observations.size
    val prevState = state.
      map { case (time, x) => mod.g(time) * x }
    val stateMean = state.map { case (t, x) => x }
    val difference = stateMean.zip(prevState).
      map { case (mt, mt1) => (mt1 - mt) * (mt1 - mt).t }.
      reduce(_ + _)

    Wishart(priorW.n + n, inv(inv(priorW.scale) + difference)).map(inv(_))
  }

  def wishartStep(
    mod:          Model, 
    priorV:       Gamma,
    priorW:       Wishart, 
    observations: Array[Data])(state: GibbsSampling.State) = {

    val latentState = GibbsSampling.sampleState(mod, observations, state.p)

    for {
      system <- sampleSystemMatrix(priorW, mod, latentState, observations)
      obs = sampleObservationMatrix(priorV, mod, latentState, observations)
    } yield GibbsSampling.State(
        Parameters(
          obs,
          system,
          state.p.m0,
          state.p.c0),
        latentState
      )
  }

  /**
    * Do some gibbs samples
    */
  def gibbsSamples(
    mod:          Model, 
    priorV:       Gamma, 
    priorW:       Wishart, 
    initParams:   Parameters, 
    observations: Array[Data]) = {

    val initState = GibbsSampling.sampleState(mod, observations, initParams)
    val init = GibbsSampling.State(initParams, initState)

    MarkovChain(init)(wishartStep(mod, priorV, priorW, observations))
  }
}
