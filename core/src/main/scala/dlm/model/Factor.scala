package dlm.core.model

import breeze.stats.distributions._
import breeze.linalg.{DenseVector, DenseMatrix}

/**
  * A DLM with a factor structure for the system matrix
  */
object FactorDlm {
  case class Parameters(
    beta: DenseMatrix[Double],
    factors: DenseVector[Double],
    dlm: DlmParameters
  )

  case class State(
    theta: Vector[SamplingState],
    p: Parameters
  )

  def sampleStep(
    priorBeta: Gaussian,
    priorFactor: Gaussian,
    priorV: InverseGamma,
    model: Dlm)(s: State): Rand[State] = ???

  def sample(
    priorBeta: Gaussian,
    priorFactor: Gaussian,
    priorV: InverseGamma,
    model: Dlm,
    ys: Vector[Data]): Process[State] = ???
}
