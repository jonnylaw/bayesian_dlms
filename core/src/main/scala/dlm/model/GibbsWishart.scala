package dlm.core.model

import breeze.stats.distributions._
import breeze.linalg.DenseMatrix

/**
  * This class learns a correlated system matrix
  *  using the InverseWishart prior on the system noise matrix
  */
object GibbsWishart {

  /**
    * Sample the system covariance matrix using an
    * Inverse Wishart prior on the system covariance matrix
    * @param priorW the prior distribution of the System evolution noise matrix
    */
  def sampleSystemMatrix(
    priorW: InverseWishart,
    g:      Double => DenseMatrix[Double],
    theta:  Vector[SamplingState]) = {

    val n = theta.size - 1

    val squaredSum = (theta.init zip theta.tail)
      .map {
        case (mt, mt1) =>
          val dt = mt1.time - mt.time
          val centered = (mt1.sample - g(dt) * mt.sample)
          (centered * centered.t) /:/ dt
      }
      .reduce(_ + _)

    val dof = priorW.nu + n
    val scale = priorW.psi + squaredSum

    InverseWishart(dof, scale)
  }

  /**
    * A single step of the Gibbs Wishart algorithm
    */
  def wishartStep(
    mod: Dlm,
    priorV: InverseGamma,
    priorW: InverseWishart,
    observations: Vector[Data]) = { s: GibbsSampling.State =>

    for {
      theta <- Smoothing.ffbsDlm(mod, observations, s.p)
      newW <- sampleSystemMatrix(priorW, mod.g, theta)
      newV <- GibbsSampling.sampleObservationMatrix(priorV,
                                                    mod.f,
                                                    observations,
                                                    theta)
    } yield GibbsSampling.State(s.p.copy(v = newV, w = newW), theta)
  }

  /**
    * Perform Gibbs Sampling using an Inverse Wishart distribution for the system
    * noise matrix
    * @param mod a DLM model specification
    * @param priorV the prior on the observation noise matrix
    * @param priorW the Inverse Wishart prior distribution on the system
    * covariance matrix
    * @param initParams the intial parameters of the Markov Chain
    * @param observations a vector of time series observations
    */
  def sample(
    mod: Dlm,
    priorV: InverseGamma,
    priorW: InverseWishart,
    initParams: DlmParameters,
    observations: Vector[Data]) = {

    val init = for {
      initState <- Smoothing.ffbs(mod, observations,
      KalmanFilter.advanceState(initParams, mod.g),
      Smoothing.step(mod, initParams.w), initParams)
    } yield GibbsSampling.State(initParams, initState)

    MarkovChain(init.draw)(wishartStep(mod, priorV, priorW, observations))
  }
}
