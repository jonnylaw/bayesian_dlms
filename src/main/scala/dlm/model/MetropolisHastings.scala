package dlm.model

import breeze.linalg.{DenseMatrix, diag, DenseVector, inv}
import breeze.stats.distributions._
import scala.math.{exp, log}
import cats.Monad
import KalmanFilter._
import Dlm._
import cats.implicits._

object MetropolisHastings {
  /**
    * State for the Metropolis Hastings algorithm
    */
  case class MhState(p: Parameters, ll: Double, accepted: Int) {
    override def toString = s"${p.toString}, $accepted"
  }

  def symmetricProposal(delta: Double)(p: Parameters): Rand[Parameters] = ???

  //   val p_logged = Parameters(p.v map log, p.w map log, p.m0, p.c0 map log)

  //   p_logged.
  //     traverse(x => Gaussian(x, delta): Rand[Double]).
  //     map(p => Parameters(p.v map exp, p.w map exp, p.m0 map exp, p.c0 map exp))
  // }

  def metropolisHastingsDlm(
    mod: Model,
    observations: Array[Data],
    proposal: Parameters => Rand[Parameters],
    initState: MhState
  ) = {

    MarkovChain(initState)(mhStep(proposal, kfLogLikelihood(mod, observations)))
  }

  def mhStep(
    proposal: Parameters => Rand[Parameters],
    likelihood: Parameters => Double
  )(state: MhState) = {

    for {
      prop_p <- proposal(state.p)
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
}
