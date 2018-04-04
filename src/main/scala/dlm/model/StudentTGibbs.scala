package dlm.model

import Dlm._
import cats.data.Kleisli
import cats.implicits._
import breeze.linalg.{DenseVector, diag, DenseMatrix}
import breeze.stats.distributions.{Rand, MarkovChain, Gamma}

object StudentT {
  /**
    * The state of the Markov chain for the Student's t-distribution
    * gibbs sampler
    */
  case class State(
    p:         Dlm.Parameters,
    variances: Vector[Double],
    state:     Vector[(Double, DenseVector[Double])]
  )

  /**
    * Sample the variances of the Normal distribution 
    * These are auxilliary variables required when calculating 
    * the one-step prediction in the Kalman Filter
    * @param ys an array of observations of length N
    * @param f the observation matrix
    * @param dof the degrees of freedom of the Student's t-distribution
    * @return a Rand distribution over the list of N variances
    */
  def sampleVariances(
    ys:    Vector[Data],
    f:     Double => DenseMatrix[Double],
    dof:   Int
  ) = { (s: State) =>

    val scale = s.p.v(0, 0)
    val alpha = (dof + 1) * 0.5
    val ft = (s.state.tail zip ys).
      map { case ((time, x), y) => 
        val fm = KalmanFilter.missingF(f, time, y.observation) 
        fm.t * x
      }

    val flatObservations = ys.
      map(_.observation).
      map(KalmanFilter.flattenObs)

    val diff = (flatObservations zip ft).
      map { case (y, fr) => (y - fr) *:* (y - fr)}.
      map(x => x(0))

    val beta = diff.map(d => (dof * scale * 0.5) + d * 0.5)

    for {
      vr <- beta traverse (b => InverseGamma(alpha, b): Rand[Double])
    } yield s.copy(variances = vr)
  }

  /**
    * Sample the (square of the) scale of the Student's t distribution
    * @param dof the degrees of freedom of the Student's t observation distribution
    * @param s the current state of the MCMC algorithm
    */
  def sampleScaleT(
    dof: Int)
    (s:  State) = {

    val t = s.variances.size
  
    val shape = t * dof * 0.5 + 1
    val rate = dof * 0.5 * s.variances.map(1.0/_).sum
    val scale = 1 / rate

    for {
      newV <- Gamma(shape, scale)
    } yield s.copy(p = s.p.copy(v = DenseMatrix(newV)))
  }

  /**
    * Sample the state, incorporating the drawn variances for each observation
    * @param mod the DLM 
    * @param observations
    */
  def sampleState(
    variances:    Vector[Double],
    mod:          Dlm.Model,
    observations: Vector[Data],
    params:       Parameters
  ) = {
    // create a list of parameters with the variance in them
    val ps = variances.map(vi => params.copy(v = DenseMatrix(vi)))

    def kalmanStep(p: Dlm.Parameters) = KalmanFilter.step(mod, p) _

    val (at, rt) = KalmanFilter.advanceState(mod.g, params.m0, 
      params.c0, 0, params.w)
    val init = KalmanFilter.initialiseState(mod, params, observations)

    // fold over the list of variances and the observations
    val filtered = (ps zip observations).
      scanLeft(init){ case (s, (p, y)) => kalmanStep(p)(s, y) }

    Rand.always(Smoothing.sample(mod, filtered, params.w).toVector)
  }

  def sampleSystemMatrix(
    mod:    Model,
    priorW: InverseGamma)
    (s:     State) = {

    val st = GibbsSampling.State(s.p, s.state)

    for {
      newS <- GibbsSampling.sampleSystemMatrix(priorW, mod.g)(st)
    } yield s.copy(p = s.p.copy(w = newS.p.w))
  }

  def stepState(
    mod:          Dlm.Model,
    observations: Vector[Data])
    (s: State) = {

    for {
      newState <- sampleState(s.variances, mod, observations, s.p)
    } yield s.copy(state = newState)
  }
  /**
    * A single step of the Student t-distribution Gibbs Sampler
    */
  def step(
    dof:    Int, 
    data:   Vector[Data],
    priorW: InverseGamma,
    mod:    Dglm.Model) = {
    val dlm = Model(mod.f, mod.g)

    Kleisli(stepState(dlm, data)) compose
      Kleisli(sampleSystemMatrix(dlm, priorW)) compose
      Kleisli(sampleVariances(data, mod.f, dof)) compose
      Kleisli(sampleScaleT(dof))
  }

  /**
    * Perform Gibbs Sampling for the Student t-distributed model
    * @param dof the degrees of freedom for the Student's t distributed 
    * observation dist
    * @param priorW the prior distribution of the system noise matrix
    * @param mod the DGLM representing the Student's t model
    * @param params the initial parameters
    */
  def sample(
    dof:    Int,
    data:   Vector[Data],
    priorW: InverseGamma,
    mod:    Dglm.Model,
    params: Dlm.Parameters
  ) = {
    val dlm = Model(mod.f, mod.g)
    val initVariances = Vector.fill(data.size)(1.0)
    val initState = sampleState(initVariances, dlm, data, params)
    val init = State(params, initVariances, initState.draw)

    MarkovChain(init)(step(dof, data, priorW, mod).run)
  }
}
