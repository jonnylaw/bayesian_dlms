package dlm.model

import Dlm._
import cats.implicits._
import breeze.linalg.{DenseVector, diag, DenseMatrix}
import breeze.stats.distributions.{Rand, MarkovChain, Gamma}

object StudentTGibbs {
  /**
    * Sample the variances of the Normal distribution 
    * These are auxilliary variables required when calculating 
    * the one-step prediction in the Kalman Filter
    * @param ys an array of observations of length N
    * @param f the observation matrix
    * @param state a sample from the state, using sampleState
    * @param dof the degrees of freedom of the Student's t-distribution
    * @param scale the (square of the) scale of the Student's t-distribution
    * @return a Rand distribution over the list of N variances
    */
  def sampleVariances(
    ys:    Vector[Data],
    f:     Double => DenseMatrix[Double],
    state: Vector[(Double, DenseVector[Double])],
    dof:   Int,
    scale: Double
  ): Rand[Vector[Double]] = {
    val alpha = (dof + 1) * 0.5

    val sortedState = state.sortBy(_._1)

    val sortedObservations = ys.sortBy(_.time).map(_.observation)

    val ft = sortedState.tail.zip(sortedObservations).
      map { case ((time, x), y) => 
        val fm = KalmanFilter.missingF(f, time, y) 
        fm.t * x
      }

    val flatObservations = sortedObservations.
      map(KalmanFilter.flattenObs)

    val diff = (flatObservations, ft).zipped.
      map { case (y, fr) => (y - fr) *:* (y - fr)}.
      map(x => x(0))

    val beta = diff.map(d => (dof * scale * 0.5) + d * 0.5)

    beta traverse (b => InverseGamma(alpha, b): Rand[Double])
  }

  /**
    * Sample the (square of the) scale of the Student's t distribution
    * @param
    */
  def sampleScaleT(
    variances: Vector[Double],
    dof:       Int): Rand[Double] = {
    val t = variances.size
  
    val shape = t * dof * 0.5 + 1
    val rate = dof * 0.5 * variances.map(1.0/_).sum
    val scale = 1 / rate

    Gamma(shape, scale)
  }

  /**
    * Sample the state, incorporating the drawn variances for each observation
    * @param 
    */
  def sampleState(
    variances:    Vector[Double],
    params:       Dlm.Parameters,
    mod:          Dlm.Model,
    observations: Vector[Data]
  ) = {
    // create a list of parameters with the variance in them
    val ps = variances.map(vi => params.copy(v = DenseMatrix(vi)))

    def kalmanStep(p: Dlm.Parameters) = KalmanFilter.step(mod, p) _

    val (at, rt) = KalmanFilter.advanceState(mod.g, params.m0, 
      params.c0, 0, params.w)

    val init = KalmanFilter.initialiseState(mod, params, observations)

    // fold over the list of variances and the observations
    val filtered = ps.zip(observations).
      scanLeft(init){ case (s, (p, y)) => kalmanStep(p)(s, y) }

    Rand.always(Smoothing.sample(mod, filtered, params.w).toVector)
  }

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
    * A single step of the Student t-distribution Gibbs Sampler
    */
  def step(
    dof:    Int, 
    data:   Vector[Data],
    priorW: InverseGamma,
    mod:    Dglm.Model
  )(state: State) = {
    val dlm = Model(mod.f, mod.g)

    for {
      latentState <- sampleState(state.variances, state.p, dlm, data)
      newW <- GibbsSampling.sampleSystemMatrix(priorW, mod.g, latentState)
      variances <- sampleVariances(data, mod.f, latentState, dof, state.p.v(0,0))
      scale <- sampleScaleT(variances, dof)
    } yield State(
      state.p.copy(v = DenseMatrix(scale), w = newW),
      variances, 
      latentState
    )
  }

  /**
    * Perform Gibbs Sampling for the Student t-distributed model
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
    val initState = sampleState(initVariances, params, dlm, data)
    val init = State(params, initVariances, initState.draw)

    MarkovChain(init)(step(dof, data, priorW, mod))
  }
}


