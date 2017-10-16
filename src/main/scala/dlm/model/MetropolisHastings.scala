package dlm.model

import breeze.linalg.{DenseMatrix, diag, DenseVector, inv}
import breeze.stats.distributions._
import scala.math.log
import breeze.numerics.exp
import cats.{Monad, Applicative}
import Dlm._
import cats.implicits._

object MetropolisHastings {
  /**
    * State for the Metropolis Hastings algorithm
    */
  case class MhState[A](parameters: A, ll: Double, accepted: Int)

  /**
    * Add a Random innovation to a numeric value using the Gaussian distribution 
    * @param delta the standard deviation of the innovation distribution
    * @param a the starting value of the parameter
    * @return a Rand[Double] representing a perturbation of the double a which can be drawn from
    */
  def proposeDouble(delta: Double)(a: Double) = 
    Gaussian(0.0, delta).map(i => a + i)

  /**
    * Simulate from a multivariate normal distribution given the cholesky decomposition of the covariance matrix
    */
  def rmvn(chol: DenseMatrix[Double])(implicit rand: RandBasis = Rand) = {
    Rand.always(chol * DenseVector.rand(chol.cols, rand.gaussian(0, 1)))
  }

  /**
    * Update the diagonal values of a covariance matrix by adding a Gaussian perturbation
    * and ensuring the resulting diagonal is symmetric
    * @param delta the standard deviation of the innovation distribution
    * @param m a diagonal DenseMatrix[Double], representing a covariance matrix
    */
  def proposeDiagonalMatrix(delta: Double)(m: DenseMatrix[Double]) = {
    rmvn(DenseMatrix.eye[Double](m.cols) * (1.0 /delta)).
      map(i => diag(m) *:* exp(i)).
      map(a => diag(a))
  }

  /**
    * Propose a new value of the parameters
    */
  def symmetricProposal(delta: Double)(p: Parameters): Rand[Parameters] = {
    val logP = Parameters(p.v.map(log), p.w.map(log), p.m0, p.c0.map(log))
    Rand.always(logP.map(x => proposeDouble(delta)(x).draw))
  }

  /**
    * Metropolis Hastings state without re-evaluating the likelihood from the previous time step
    */
  def mhStep[A](
    proposal: A => Rand[A],
    likelihood: A => Double
  )(state: MhState[A]) = {

    for {
      prop_p <- proposal(state.parameters)
      prop_ll = likelihood(prop_p)
      a = prop_ll - state.ll
      u <- Uniform(0, 1)
      next = if (log(u) < a) {
        MhState(prop_p, prop_ll, state.accepted + 1)
      } else {
        state
      }
    } yield next
  }

  /**
    * Run metropolis hastings for a DLM, using the kalman filter to calculate the likelihood
    */
  def metropolisHastingsDlm(
    mod: Model,
    observations: Array[Data],
    proposal: Parameters => Rand[Parameters],
    initP: Parameters
  ) = {
    val initState = MhState[Parameters](initP, -1e99, 0)
    MarkovChain(initState)(mhStep[Parameters](proposal, KalmanFilter.logLikelihood(mod, observations)))
  }
}
