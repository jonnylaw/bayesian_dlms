package dlm.model

import breeze.stats.distributions._
import breeze.linalg._
import Dlm._

/**
  * This class learns a correlated system matrix using the InverseWishart prior on the system noise matrix
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

    GibbsSampling.sampleObservationMatrix(priorV, mod, state, observations)
  }

  /**
    * Sample the system covariance matrix using a Wishart prior on the system precision matrix
    */
  def sampleSystemMatrix(
    priorW:       InverseWishart,
    mod:          Model, 
    state:        Array[(Time, DenseVector[Double])]): Rand[DenseMatrix[Double]] = {

    val n = state.size - 1
    val prevState = state.map { case (time, x) => mod.g(time) * x }
    val stateMean = state.map { case (t, x) => x }.tail
    val difference = stateMean.zip(prevState).
      map { case (mt, mt1) => (mt - mt1) * (mt - mt1).t }.
      reduce(_ + _)

    val dof = priorW.nu + n
    val scale = priorW.psi + difference

    InverseWishart(dof, scale)
  }

  def wishartStep(
    mod:          Model, 
    priorV:       Gamma,
    priorW:       InverseWishart, 
    observations: Array[Data])(state: GibbsSampling.State) = {

    val latentState = GibbsSampling.sampleState(mod, observations, state.p)

    for {
      system <- sampleSystemMatrix(priorW, mod, latentState)
      obs <- sampleObservationMatrix(priorV, mod, latentState, observations)
      p = Parameters(obs, system, state.p.m0, state.p.c0)
    } yield GibbsSampling.State(p, latentState)
  }

  /**
    * Do some gibbs samples
    */
  def gibbsSamples(
    mod:          Model, 
    priorV:       Gamma, 
    priorW:       InverseWishart, 
    initParams:   Parameters, 
    observations: Array[Data]) = {

    val initState = GibbsSampling.sampleState(mod, observations, initParams)
    val init = GibbsSampling.State(initParams, initState)

    MarkovChain(init)(wishartStep(mod, priorV, priorW, observations))
  }
}
