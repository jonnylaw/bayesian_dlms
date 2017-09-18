import breeze.linalg.{DenseMatrix, diag, DenseVector, inv}
import breeze.stats.distributions._
import scala.math.{exp, log}
import cats.Monad
import KalmanFilter._
import Dlm._
import cats.implicits._

object MetropolisHastings {
  def symmetric_proposal(delta: Double)(p: Parameters): Rand[Parameters] = {
    p.traverse(x => Gaussian(x, delta): Rand[Double])
  }

  def metropolis_hastings_dlm(
    mod: Model,
    observations: Array[Data],
    proposal: Parameters => Rand[Parameters],
    initState: MhState
  ) = {

    MarkovChain(initState)(mh_step(proposal, kf_ll(mod, observations)))
  }

  /**
    * State for the Metropolis Hastings algorithm
    */
  case class MhState(p: Parameters, ll: Double, accepted: Int)

  def mh_step(
    proposal: Parameters => Rand[Parameters],
    likelihood: Parameters => Double
  )(state: MhState) = {

    for {
      prop_p <- proposal(state.p)
      prop_ll = likelihood(prop_p)
      a = prop_ll - state.ll
      u <- Uniform(0, 1)
      next = if (a < log(u)) {
        MhState(prop_p, prop_ll, state.accepted + 1)
      } else {
        state
      }
    } yield next
  }
}
