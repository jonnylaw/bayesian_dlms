package dlm.model

import Dglm._
import ParticleGibbs.State
import breeze.stats.distributions.{Multinomial, Rand}
import breeze.linalg.DenseVector
import breeze.stats.mean
import math.{log, exp}
import cats.Functor
import cats.implicits._

/**
  * Particle Gibbs with Ancestor Sampling
  * Requires a Tractable state evolution kernel
  */
object ParticleGibbsAncestor extends App {

  /**
    * Calculate the probability of each particle at time t-1 evolving to the 
    * conditioned particle at time t
    */
  def transitionProbability(
    sampledStates:    Vector[DenseVector[Double]],
    weights:          Vector[Double],
    conditionedState: DenseVector[Double],
    time:             Time,
    mod:              Model,
    p:                Dlm.Parameters
  ): Vector[Double] = {

    val res = for {
      (x, w) <- (sampledStates, weights).zipped
      ll = MultivariateGaussianSvd(mod.g(time) * x, p.w).logPdf _
    } yield w * ll(conditionedState)

    res.toVector
  }

  /**
    * Sample n items with replacement from xs with probability ws
    */
  def sample[A](n: Int, xs: Vector[A], ws: Vector[Double]): Vector[A] = {
    val indices = Multinomial(DenseVector(ws.toArray)).sample(n)

    indices.map(xs(_)).toVector
  }

  def sampleAllStates(n: Int, xs: List[LatentState], w: List[Double]): List[LatentState] = {
    val is = sample(n, Vector.range(0, n), w.toVector).toList
    is map (i => xs.map(state => state(i)))
  }

  def step(
    mod: Model,
    p:   Dlm.Parameters
  ) = (s: State, a: (Data, DenseVector[Double])) => a match {

    case (Data(time, Some(y)), conditionedState) =>
      val n = s.states.head.size

      // sample n-1 particles from ALL of the states
      val x = sampleAllStates(n-1, s.states, s.weights)

      // calculate transition probability
      val transProb = transitionProbability(x.head.map(_._2).toVector, s.weights.toVector, conditionedState, time, mod, p)

      // sample the nth particle proportional to transProb
      val xn = sample(1, s.states.toVector, transProb)

      // advance the n-1 states
      val x1 = ParticleFilter.advanceState(mod, time, x.head.map(_._2), p).draw

      // lump states back together
      val allState = (conditionedState :: x1)

      // calculate the weights
      val w = ParticleFilter.calcWeights(mod, time, allState, y, p)

      // log-sum-exp and calculate log-likelihood
      val max = w.max
      val w1 = w map (a => exp(a - max))
      val ll = s.ll + max + log(mean(w1))

      State(allState.map(u => (time, u)) :: x, w1, ll)

    case (Data(time, None), conditionedState) => throw new Exception
      // val n = s.states.head.size
      // // sample n-1 particles from ALL of the states
      // val x = sample(n-1, s.states.transpose.toVector, s.weights.toVector)

      // // calculate transition probability
      // val transProb = transitionProbability(x.head, s.weights.toVector, conditionedState, time, mod, p)

      // // sample the nth particle proportional to transProb
      // val xn = sample(1, s.states.toVector, transProb)

      // // advance the n-1 states
      // val x1 = ParticleFilter.advanceState(mod, time, x.head, p)

      // // lump states back together
      // val allState = (conditionedState :: x1)

      // State(allState.map(x => (time x)) :: s.states, w1, s.ll)
  }

  /**
    * Run Particle Gibbs with Ancestor Sampling
    * @param n the number of particles to have in the sampler
    * @param mod the DGLM model specification
    * @param p the model parameters
    * @param obs a list of measurements
    * @param state the state which is to be conditioned upon
    */
  def filter(
    n:   Int,
    mod: Model,
    p:   Dlm.Parameters,
    obs: List[Data])(state: LatentState): Rand[(Double, LatentState)] = {

    val firstTime = obs.map(d => d.time).min
    val x0 = ParticleGibbs.initState(p).sample(n-1).toList.map(x => (firstTime - 1, x))
    val init = State(List(x0), List.fill(n - 1)(1.0 / n), 0.0)

    val filtered = (obs, state.map(_._2)).
      zipped.
      foldLeft(init)(step(mod, p))

    ParticleGibbs.sampleState(filtered.states, filtered.weights) map ((filtered.ll, _))
  }
}
