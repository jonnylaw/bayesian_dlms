package dlm.model

import breeze.stats.distributions._
import breeze.linalg._
import Dlm._

/**
  * This class learns a correlated system matrix using the InverseWishart prior on the system noise matrix
  */
object GibbsWishart {

  /**
    * Sample the system covariance matrix using an Inverse Wishart prior on the system covariance matrix
    */
  def sampleSystemMatrix(
    priorW: InverseWishart,
    mod:    Model, 
    state:  Array[(Time, DenseVector[Double])]) = {

    val n = state.size - 1
    val prevState = state.init.map { case (time, x) => mod.g(time) * x }
    val stateMean = state.tail.map { case (t, x) => x }
    val difference = stateMean.zip(prevState).
      map { case (mt, mt1) => (mt - mt1) * (mt - mt1).t }.
      reduce(_ + _)

    val dof = priorW.nu + n
    val scale = priorW.psi + difference

    InverseWishart(dof, scale)
  }

  /**
    * A single step of the Gibbs Wishart algorithm
    */
  def wishartStep(
    mod:          Model, 
    priorV:       InverseGamma,
    priorW:       InverseWishart, 
    observations: Array[Data])(state: GibbsSampling.State) = {

    for {
      system <- sampleSystemMatrix(priorW, mod, state.state)
      latentState = GibbsSampling.sampleState(mod, observations, Parameters(state.p.v, system, state.p.m0, state.p.c0))
      obs <- GibbsSampling.sampleObservationMatrix(
        priorV, mod.f, latentState, observations)
      p = Parameters(obs, system, state.p.m0, state.p.c0)
    } yield GibbsSampling.State(p, latentState)
  }

  /**
    * Do some gibbs samples
    * @param mod a DLM model specification
    * @param priorV the prior on the observation noise matrix
    * @param priorW the prior distribution on the system covariance matrix
    * @param initParams 
    */
  def sample(
    mod:          Model, 
    priorV:       InverseGamma, 
    priorW:       InverseWishart, 
    initParams:   Parameters, 
    observations: Array[Data]) = {

    val initState = GibbsSampling.sampleState(mod, observations, initParams)
    val init = GibbsSampling.State(initParams, initState)

    MarkovChain(init)(wishartStep(mod, priorV, priorW, observations))
  }
}
