package dlm.core.model

import breeze.linalg.{DenseMatrix, diag, DenseVector}
import breeze.stats.distributions._
import scala.math.log
import breeze.numerics.exp
import cats.{Applicative, Traverse}
import cats.implicits._

object Metropolis {

  /**
    * State for the Metropolis algorithm
    * @param
    */
  case class State[A](parameters: A, ll: Double, accepted: Int)

  /**
    * Add a Random innovation to a numeric value using the Gaussian distribution
    * @param delta the standard deviation of the innovation distribution
    * @param a the starting value of the Double
    * @return a Rand[Double] representing a perturbation of the
    * double a which can be drawn from
    */
  def proposeDouble(delta: Double)(a: Double) =
    Gaussian(0.0, delta).map(i => a + i)

  /**
    * Add a Random innovation to a DenseVector[Double] using
    * the Gaussian distribution
    * @param delta the standard deviation of the innovation distribution
    * @param a the starting value of the parameter
    * @return a Rand[DenseVector[Double]] representing a perturbation
    *  of the double a which can be drawn from
    */
  def proposeVector(delta: Double)(a: DenseVector[Double]) = {
    for {
      i <- Applicative[Rand].replicateA(a.size, Gaussian(0.0, delta))
    } yield DenseVector(i.toArray) + a
  }

  /**
    * Simulate from a multivariate normal distribution
    * given the cholesky decomposition of the covariance matrix
    */
  def rmvn(chol: DenseMatrix[Double])(implicit rand: RandBasis = Rand) = {
    Rand.always(chol * DenseVector.rand(chol.cols, rand.gaussian(0, 1)))
  }

  /**
    * Update the diagonal values of a covariance matrix by adding a Gaussian perturbation
    * and ensuring the resulting diagonal is symmetric
    * @param delta the standard deviation of the innovation distribution
    * @param m a diagonal DenseMatrix[Double], representing a covariance matrix
    * @return a distribution over the diagonal matrices
    */
  def proposeDiagonalMatrix(delta: Double)(m: DenseMatrix[Double]) = {
    rmvn(DenseMatrix.eye[Double](m.cols) * (math.sqrt(delta)))
      .map(i => diag(m) *:* exp(i))
      .map(a => diag(a))
  }

  /**
    * Propose a new value of the parameters on the log scale
    */
  def symmetricProposal(delta: Double)(
      p: DlmParameters): Rand[DlmParameters] = {
    for {
      v <- proposeDiagonalMatrix(delta)(p.v)
      w <- proposeDiagonalMatrix(delta)(p.w)
      m0 <- proposeVector(delta)(p.m0)
      c0 <- proposeDiagonalMatrix(delta)(p.c0)
    } yield DlmParameters(v, w, m0, c0)
  }

  /**
    * A Single Step without acceptance ratio
    * this requires re-evaluating the likelihood at each step
    */
  def step[A](
      proposal: A => Rand[A],
      prior: A => Double,
      likelihood: A => Double)(state: (A, Double)) = {

    MarkovChain.Kernels.metropolis(proposal)((a: A) => prior(a) + likelihood(a))
  }

  /**
    * Metropolis kernel without re-evaluating the likelihood
    * from the previous time step and keeping track of the acceptance ratio
    */
  def mStep[A](
    proposal: A => Rand[A],
    prior: A => Double,
    likelihood: A => Double)(state: State[A]) = {

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

  def mAccept[A](
    proposal: A => Rand[A],
    logMeasure: A => Double)(param: A) = {

    for {
      propP <- proposal(param)
      a = logMeasure(propP) - logMeasure(param)
      u <- Uniform(0, 1)
      next = if (log(u) < a) {
        (propP, 1)
      } else {
        (param, 0)
      }
    } yield next
  }

  /**
    * Run the metropolis algorithm for a DLM
    * using the kalman filter to calculate the likelihood
    */
  def dlm[T[_]: Traverse](
      mod: Dlm,
      observations: T[Data],
      proposal: DlmParameters => Rand[DlmParameters],
      prior: DlmParameters => Double,
      initP: DlmParameters
  ) = {
    val initState = State[DlmParameters](initP, -1e99, 0)
    val ll = (p: DlmParameters) =>
      KalmanFilter.likelihood(mod, observations)(p)

    MarkovChain(initState)(mStep[DlmParameters](proposal, prior, ll))
  }

  /**
    * Use particle marginal metropolis algorithm for a DGLM model
    */
  def dglm[T[_]: Traverse](
    mod: DglmModel,
    observations: T[Data],
    proposal: DlmParameters => Rand[DlmParameters],
    prior: DlmParameters => Double,
    initP: DlmParameters,
    n: Int) = {

    val initState = State[DlmParameters](initP, -1e99, 0)
    val ll = (p: DlmParameters) =>
      ParticleFilter.likelihood(mod, observations, n)(p)

    MarkovChain(initState)(mStep[DlmParameters](proposal, prior, ll))
  }
}

object MetropolisHastings {
  def mhStep[A](
      proposal: A => ContinuousDistr[A],
      prior: A => Double,
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

  def mhAccept[A](
    proposal: A => ContinuousDistr[A],
    logMeasure: A => Double)(param: A) = {

    for {
      propP <- proposal(param)
      nextp = proposal(param).logPdf(propP)
      lastp = proposal(propP).logPdf(param)
      a = logMeasure(propP) - nextp - logMeasure(param) + lastp
      u <- Uniform(0, 1)
      next = if (log(u) < a) {
        (propP, 1)
      } else {
        (param, 0)
      }
    } yield next
  }

  /**
    * Run Metropolis-Hastings algorithm for a DLM, using the kalman filter to calculate the likelihood
    */
  def dlm(mod: Dlm,
          observations: Vector[Data],
          proposal: DlmParameters => ContinuousDistr[DlmParameters],
          prior: DlmParameters => Double,
          initP: DlmParameters) = {

    val ll = (p: DlmParameters) =>
      KalmanFilter.likelihood(mod, observations)(p)

    val initState = Metropolis.State[DlmParameters](initP, -1e99, 0)
    MarkovChain(initState)(mhStep[DlmParameters](proposal, prior, ll))
  }

  /**
    * Particle Marginal Metropolis Hastings for a ContinuousTime Model
    * Where the log-likelihood is an estimate calculated using the bootstrap
    * particle filter
    * @param mod a DGLM model
    * @param observations an array of observations
    * @param a proposal distribution for the parameters
    * @param initP the intial parameters to start the Markov Chain
    * @param n the number of particles in the PF
    * @return a Markov Chain Process which can be drawn from
    */
  def pmmh(mod: DglmModel,
           observations: Vector[Data],
           proposal: DlmParameters => ContinuousDistr[DlmParameters],
           prior: DlmParameters => Double,
           initP: DlmParameters,
           n: Int) = {

    val ll = (p: DlmParameters) =>
      ParticleFilter.likelihood(mod, observations, n)(p)

    val initState = Metropolis.State[DlmParameters](initP, -1e99, 0)

    MarkovChain(initState)(mhStep[DlmParameters](proposal, prior, ll))
  }
}
