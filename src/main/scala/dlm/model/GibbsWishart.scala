package dlm.model

import breeze.stats.distributions._
import breeze.linalg.{DenseMatrix, DenseVector}
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
    g:      Double => DenseMatrix[Double], 
    state:  Vector[(Double, DenseVector[Double])]) = {

    val n = state.size - 1
    val sortedState = state.sortBy(_._1)
    val times = sortedState.map(_._1)
    val deltas = GibbsSampling.diff(times)
    val advanceState = (deltas, sortedState.init.map(_._2)).
      zipped.
      map { case (dt, x) => g(dt) * x }

    val stateMean = sortedState.map(_._2).tail

    val squaredSum = (deltas zip stateMean zip advanceState).
      map { case ((dt, mt), at) => (mt - at) * (mt - at).t /:/ dt }.
      reduce(_ + _)

    val dof = priorW.nu + n
    val scale = priorW.psi + squaredSum

    InverseWishart(dof, scale)
  }

  /**
    * A single step of the Gibbs Wishart algorithm
    */
  def wishartStep(
    mod:          Model, 
    priorV:       InverseGamma,
    priorW:       InverseWishart, 
    observations: Vector[Data])(state: GibbsSampling.State) = {

    for {
      system <- sampleSystemMatrix(priorW, mod.g, state.state)
      latentState <- Smoothing.ffbs(mod, observations, state.p.copy(w = system))
      obs <- GibbsSampling.sampleObservationMatrix(priorV, mod.f, latentState, observations)
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
    observations: Vector[Data]) = {

    val initState = Smoothing.ffbs(mod, observations, initParams).draw
    val init = GibbsSampling.State(initParams, initState)

    MarkovChain(init)(wishartStep(mod, priorV, priorW, observations))
  }
}
