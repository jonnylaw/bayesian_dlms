package com.github.jonnylaw.dlm

import breeze.linalg.DenseVector
import breeze.stats.distributions.{Multinomial, Rand}
import cats.implicits._
import math.{exp, log}
import breeze.stats.mean
import ParticleFilter._

case class PgState(conditionedState: Map[Double, DenseVector[Double]],
                   states: Vector[Vector[(Double, DenseVector[Double])]],
                   weights: Vector[Double],
                   ll: Double)

/**
  * Particle Gibbs Sampler for A Dynamic Generalised Linear Dglm
  */
case class ParticleGibbs(n: Int) {

  def initialConditionedState(model: Dglm,
                              p: DlmParameters,
                              ys: Vector[Data]) = {
    val n0 = math.floor(n / 5).toInt
    val st = ParticleFilter(n, n0, multinomialResample).filter(model, ys, p)
    val ws = st.map(_.weights).last
    val states = st.map(d => d.state.map((d.time, _)))

    ParticleGibbs.sampleState(states, ws).draw.toMap
  }

  def initialiseState(model: Dglm, p: DlmParameters, ys: Vector[Data]) = {

    val t0 = ys.foldLeft(0.0)((t0, d) => math.min(t0, d.time))
    val x0 = MultivariateGaussianSvd(p.m0, p.c0)
      .sample(n - 1)
      .map(x => (t0, x))
      .toVector

    // sample the first conditioned state
    val conditionedState = initialConditionedState(model, p, ys)

    PgState(conditionedState, Vector(x0), Vector.fill(n - 1)(1.0 / n), 0.0)
  }

  def step(mod: Dglm, p: DlmParameters)(s: PgState, d: Data): PgState = {

    val y = KalmanFilter.flattenObs(d.observation)
    val dt = d.time - s.states.last.head._1

    // resample using the previous weights
    val resampledX = multinomialResample(s.states.head, s.weights)

    // advance n-1 states from time t, located at the head of the list
    val x1 =
      advanceState(dt, resampledX.map(_._2), mod, p).draw.map(x => (d.time, x))

    if (y.data.isEmpty) {
      PgState(s.conditionedState,
              s.states :+ x1,
              Vector.fill(x1.size - 1)(1.0 / x1.size),
              s.ll)
    } else {
      // concat conditioned state at current time and advanced state
      val cond: (Double, DenseVector[Double]) =
        (d.time, s.conditionedState.getOrElse(d.time, x1.head._2))
      val x = cond +: x1

      // calculate weights of all n states at time t
      val w = calcWeights(mod, d.time, x.map(_._2), d.observation, p)

      // calculate updated pseudo log-likelihood
      val max = w.max
      val w1 = w map (a => exp(a - max))
      val ll = s.ll + max + log(mean(w1))

      PgState(s.conditionedState, s.states :+ x1, w1.tail, ll)
    }
  }

  /**
    * Perform the PG filter
    * @param n the total number of particles in the filter
    * @param n0 if ESS < n0 then resample
    */
  def filter(model: Dglm, ys: Vector[Data], p: DlmParameters): PgState = {

    val init = initialiseState(model, p, ys)
    ys.foldLeft(init)(step(model, p))
  }
}

object ParticleGibbs {

  /**
    * Using the weights at time T (the end of all observations) sample a
    * path from the collection of paths
    * @param states a collection of paths with ancestory, the outer list is
    * of length T, theinner length N
    * @param weights particle weights at time T
    * @return a single path
    */
  def sampleState(
      states: Vector[Vector[(Double, DenseVector[Double])]],
      weights: Vector[Double]): Rand[Vector[(Double, DenseVector[Double])]] = {
    for {
      k <- Multinomial(DenseVector(weights.toArray))
      x = states.transpose
    } yield x(k)
  }

  /**
    * Sample the conditioned state from the Particle Gibbs Sampler
    * @param n the number of particles to use in the particle filter
    */
  def sample(n: Int, mod: Dglm, ys: Vector[Data], p: DlmParameters) = {
    val filtered = ParticleGibbs(n).filter(mod, ys, p)
    sampleState(filtered.states, filtered.weights) map ((filtered.ll, _))
  }
}
