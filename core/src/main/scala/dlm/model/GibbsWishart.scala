package dlm.core.model

import breeze.stats.distributions._
import breeze.linalg.{DenseMatrix, DenseVector}

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
  def sampleSystemMatrix(priorW: InverseWishart,
                         g: Double => DenseMatrix[Double],
                         theta: Vector[(Double, DenseVector[Double])]) = {

    val n = theta.size - 1

    val squaredSum = (theta.init zip theta.tail)
      .map {
        case (mt, mt1) =>
          val dt = mt1._1 - mt._1
          val centered = (mt1._2 - g(dt) * mt._2)
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
  def wishartStep(mod: Dlm,
                  priorV: InverseGamma,
                  priorW: InverseWishart,
                  observations: Vector[Data]) = { s: GibbsSampling.State =>
    for {
      theta <- Smoothing.ffbsDlm(mod, observations, s.p)
      st = theta.map(a => (a.time, a.sample))
      newW <- sampleSystemMatrix(priorW, mod.g, st.toVector)
      newV <- GibbsSampling.sampleObservationMatrix(priorV,
                                                    mod.f,
                                                    observations,
                                                    st.toVector)
    } yield GibbsSampling.State(s.p.copy(v = newV, w = newW), st.toVector)
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
  def sample(mod: Dlm,
             priorV: InverseGamma,
             priorW: InverseWishart,
             initParams: DlmParameters,
             observations: Vector[Data]) = {

    val init = for {
      initState <- Smoothing.ffbs(mod, observations,
      KalmanFilter.advanceState(initParams, mod.g),
      Smoothing.step(mod, initParams.w), initParams)
      st = initState.map(a => (a.time, a.sample))
    } yield GibbsSampling.State(initParams, st)

    MarkovChain(init.draw)(wishartStep(mod, priorV, priorW, observations))
  }
}
