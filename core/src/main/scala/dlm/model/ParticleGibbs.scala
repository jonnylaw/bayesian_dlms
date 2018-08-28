package core.dlm.model

import breeze.linalg.DenseVector
import breeze.stats.distributions.{Multinomial, Rand}
import cats.implicits._
import cats.Traverse
import math.{exp, log}
import breeze.stats.mean
import ParticleFilter._
import Dlm.Data

case class PgState(
  conditionedState: Map[Double, DenseVector[Double]],
  states: List[List[(Double, DenseVector[Double])]],
  weights: List[Double],
  ll: Double)

/**
  * Particle Gibbs Sampler for A Dynamic Generalised Linear DglmModel
  */
object ParticleGibbs {
  def initialiseState[T[_]: Traverse](
    n: Int,
    model: DglmModel,
    p: DlmParameters,
    ys: T[Data]) = {

    val t0 = ys.foldLeft(0.0)((t0, d) => math.min(t0, d.time))
    val x0 = MultivariateGaussianSvd(p.m0, p.c0)
      .sample(n - 1)
      .toList
      .map(x => (t0 - 1.0, x))

    // run the PF to sample the first conditioned state
    val st = ParticleFilter(n).filterTraverse(model, ys, p)
    val ws = st.map(_.weights).toList.last
    val states = st.map(d => d.state.map((d.time, _)).toList).toList
    val conditionedState =
      ParticleGibbs.sampleState(states, ws.toList).draw.toMap

    PgState(conditionedState, List(x0), List.fill(n - 1)(1.0 / n), 0.0)
  }

  def step(mod: DglmModel, p: DlmParameters)(s: PgState, d: Data): PgState = {

    val y = KalmanFilter.flattenObs(d.observation)
    // resample using the previous weights
    val resampledX = multinomialResample(s.states.head.toVector, s.weights.toVector)

    // advance n-1 states from time t, located at the head of the list
    val x1 =
      advanceState(mod, d.time, resampledX.map(_._2).toList, p).draw.map(x =>
        (d.time, x))

    if (y.data.isEmpty) {
      PgState(s.conditionedState,
              x1 :: s.states,
              List.fill(x1.size - 1)(1.0 / x1.size),
              s.ll)
    } else {
      // concat conditioned state at current time and advanced state
      val cond: (Double, DenseVector[Double]) =
        (d.time, s.conditionedState.getOrElse(d.time, x1.head._2))
      val x = cond :: x1

      // calculate weights of all n states at time t
      val w = calcWeights(mod, d.time, x.map(_._2), d.observation, p)

      // calculate updated pseudo log-likelihood
      val max = w.max
      val w1 = w map (a => exp(a - max))
      val ll = s.ll + max + log(mean(w1))

      PgState(s.conditionedState, x1 :: s.states, w1.tail, ll)
    }
  }

  /**
    * Using the weights at time T (the end of all observations) sample a path from the collection
    * of paths
    * @param states a collection of paths with ancestory, the outer list is of length T, inner length N
    * @param weights particle weights at time T
    * @return a single path
    */
  def sampleState(
      states: List[List[(Double, DenseVector[Double])]],
      weights: List[Double]): Rand[List[(Double, DenseVector[Double])]] = {
    for {
      k <- Multinomial(DenseVector(weights.toArray))
      x = states.transpose
    } yield x(k)
  }

  /**
    * Perform the PG filter
    */
  def filter[T[_]: Traverse](n: Int,
                             model: DglmModel,
                             ys: T[Dlm.Data],
                             p: DlmParameters): PgState = {

    val init = initialiseState(n, model, p, ys)
    ys.foldLeft(init)(step(model, p))
  }

  /**
    * Sample the conditioned state from the Particle Gibbs Sampler
    */
  def sample(n: Int, p: DlmParameters, mod: DglmModel, ys: List[Data]) = {

    val filtered = filter(n, mod, ys, p)
    sampleState(filtered.states, filtered.weights) map ((filtered.ll, _))
  }
}
