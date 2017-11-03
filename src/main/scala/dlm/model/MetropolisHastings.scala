package dlm.model

import breeze.linalg.{DenseMatrix, diag, DenseVector, inv}
import breeze.stats.distributions._
import scala.math.log
import breeze.numerics.exp
import cats.{Monad, Applicative}
import Dlm._
import cats.implicits._

object Metropolis {
  /**
    * State for the Metropolis algorithm
    */
  case class State[A](parameters: A, ll: Double, accepted: Int)

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
    * Metropolis kernel without re-evaluating the likelihood from the previous time step
    */
  def mStep[A](
    proposal:   A => Rand[A],
    prior:      A => Double,
    likelihood: A => Double
  )(state: State[A]) = {

    for {
      propP <- proposal(state.parameters)
      prop_ll = likelihood(propP) + prior(propP)
      a = prop_ll - state.ll
      u <- Uniform(0, 1)
      next = if (log(u) < a) {
        State(propP, prop_ll, state.accepted + 1)
      } else {
        state
      }
    } yield next
  }

  /**
    * Run the metropolis algorithm for a DLM, using the kalman filter to calculate the likelihood
    */
  def dlm(
    mod:          Model,
    observations: Array[Data],
    proposal:     Parameters => Rand[Parameters],
    prior:        Parameters => Double,
    initP:        Parameters
  ) = {
    val initState = State[Parameters](initP, -1e99, 0)
    MarkovChain(initState)(mStep[Parameters](proposal, 
      prior, KalmanFilter.logLikelihood(mod, observations)))
  }


  def dglm(
    mod:          Dglm.Model,
    observations: Array[Data],
    proposal:     Parameters => Rand[Parameters],
    prior:        Parameters => Double,
    initP:        Parameters,
    n:            Int
  ) = {
    val initState = State[Parameters](initP, -1e99, 0)
    MarkovChain(initState)(mStep[Parameters](proposal, 
      prior, ParticleFilter.likelihood(mod, observations, n)))
  }

}

object MetropolisHastings {
  def mhStep[A](
    proposal:   A => ContinuousDistr[A],
    prior:      A => Double,
    likelihood: A => Double
  )(state: Metropolis.State[A]) = {

    for {
      propP <- proposal(state.parameters)
      nextp = proposal(state.parameters).logPdf(propP)
      lastp = proposal(propP).logPdf(state.parameters)
      propll = likelihood(propP) + prior(propP)
      a = propll - nextp - state.ll + lastp
      u <- Uniform(0, 1)
      next = if (log(u) < a) {
        Metropolis.State(propP, propll, state.accepted + 1)
      } else {
        state
      }
    } yield next
  }

  /**
    * Run Metropolis-Hastings algorithm for a DLM, using the kalman filter to calculate the likelihood
    */
  def dlm(
    mod:          Model,
    observations: Array[Data],
    proposal:     Parameters => ContinuousDistr[Parameters],
    prior:        Parameters => Double,
    initP:        Parameters
  ) = {
    val initState = Metropolis.State[Parameters](initP, -1e99, 0)
    MarkovChain(initState)(mhStep[Parameters](proposal,
      prior, KalmanFilter.logLikelihood(mod, observations)))
  }

  /**
    * Particle Marginal Metropolis Hastings for a DGLM
    * Where the log-likelihood is an estimate calculated using the bootstrap
    * particle filter 
    * @param mod a DGLM model 
    * @param observations an array of observations
    * @param a proposal distribution for the parameters
    * @param initP the intial parameters to start the Markov Chain
    * @param n the number of particles in the PF
    * @return a Markov Chain Process which can be drawn from
    */
  def pmmh(
    mod:          Dglm.Model,
    observations: Array[Data],
    proposal:     Parameters => ContinuousDistr[Parameters],
    prior:        Parameters => Double,
    initP:        Parameters,
    n:            Int
  ) = {
    val initState = Metropolis.State[Parameters](initP, -1e99, 0)
    MarkovChain(initState)(mhStep[Parameters](proposal, 
      prior, ParticleFilter.likelihood(mod, observations, n)))
  }

}
