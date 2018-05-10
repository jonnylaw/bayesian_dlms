package dlm.model

import breeze.stats.distributions._
import breeze.linalg.DenseMatrix
import cats.data.Kleisli
import Dlm._

/**
  * This class learns a correlated system matrix
  *  using the InverseWishart prior on the system noise matrix
  */
object GibbsWishart {

  /**
    * Sample the system covariance matrix using an Inverse Wishart prior on the system covariance matrix
    */
  def sampleSystemMatrix(
    priorW: InverseWishart,
    g:      Double => DenseMatrix[Double])
    (s:     GibbsSampling.State) = {

    val n = s.state.size - 1
    val stateMean = s.state

    val squaredSum = (stateMean.init zip stateMean.tail).
      map { case (mt, mt1) =>
        val dt = mt1._1 - mt._1
        val centered = (mt1._2 - g(dt) * mt._2)
        (centered * centered.t) /:/ dt }.
      reduce(_ + _)

    val dof = priorW.nu + n
    val scale = priorW.psi + squaredSum

    for {
      newW <- InverseWishart(dof, scale)
    } yield s.copy(p = s.p.copy(w = newW))
  }

  /**
    * A single step of the Gibbs Wishart algorithm
    */
  def wishartStep(
    mod:          Model, 
    priorV:       InverseGamma,
    priorW:       InverseWishart, 
    observations: Vector[Data]) = {

    Kleisli(sampleSystemMatrix(priorW, mod.g)) compose
      Kleisli(GibbsSampling.ffbs(mod, observations)) compose
      Kleisli(GibbsSampling.sampleObservationMatrix(priorV, mod.f, observations))
  }

  /**
    * Perofrm Gibbs Sampling using an Inverse Wishart distribution for the system
    * noise matrix
    * @param mod a DLM model specification
    * @param priorV the prior on the observation noise matrix
    * @param priorW the Inverse Wishart prior distribution on the system
    * covariance matrix
    * @param initParams the intial parameters of the Markov Chain
    * @param observations a vector of time series observations
    */
  def sample(
    mod:          Model, 
    priorV:       InverseGamma, 
    priorW:       InverseWishart, 
    initParams:   Parameters, 
    observations: Vector[Data]) = {

    val initState = Smoothing.ffbs(mod, observations, initParams).draw
    val init = GibbsSampling.State(initParams, initState)

    MarkovChain(init)(wishartStep(mod, priorV, priorW, observations).run)
  }
}
