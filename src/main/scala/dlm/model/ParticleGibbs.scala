package dlm.model

import Dglm._
import breeze.linalg.DenseVector
import breeze.stats.distributions.{Multinomial, Rand}
import cats.implicits._
import math.{exp, log}
import breeze.stats.mean
import ParticleFilter._
import Dlm.Data

/**
  * Particle Gibbs Sampler for A Dynamic Generalised Linear Model
  */
object ParticleGibbs {
  case class State(
    states:  List[List[(Double, DenseVector[Double])]],
    weights: List[Double],
    ll:      Double)

  def initState(p: Dlm.Parameters) = {
    MultivariateGaussianSvd(p.m0, p.c0)
  }

  def step(
    mod: Model,
    p:   Dlm.Parameters) = (s: State, a: (Data, DenseVector[Double])) => a match {

    case (d, conditionedState) =>

      val y = KalmanFilter.flattenObs(d.observation)

      if (y.data.isEmpty) {
        // resample using the previous weights
        val resampledX = resample(s.states.head.toVector, s.weights.toVector)

        // advance n-1 states from time t, located at the head of the list
        val x1 = advanceState(mod.g, d.time, s.states.head.map(_._2), p).draw

        State(x1.map(x => (d.time, x)) :: s.states, List.fill(x1.size - 1)(1.0 / x1.size), s.ll)
      } else {
        // resample using the previous weights
        val resampled = resample(s.states.head.toVector, s.weights.toVector).toList

        // advance n-1 resampled states from time t
        val x1 = advanceState(mod.g, d.time, resampled.map(_._2), p).draw

        // concat conditioned state and advanced state
        val x: List[DenseVector[Double]] = (conditionedState :: x1)

        // calculate weights of all n states at time t
        val w = calcWeights(mod, d.time, x, d.observation, p)

        // log-sum-exp and calculate log-likelihood
        val max = w.max
        val w1 = w map (a => exp(a - max))
        val ll = s.ll + max + log(mean(w1))

        State(x1.map(x => (d.time, x)) :: s.states, w1.tail, ll)
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
    states:  List[List[(Double, DenseVector[Double])]], 
    weights: List[Double]): Rand[List[(Double, DenseVector[Double])]]  = {
    for {
      k <- Multinomial(DenseVector(weights.toArray))
      x = states.transpose
    } yield x(k)
  }

  /**
    * Run the Particle Gibbs Filter, given a samples value of the state
    * @param n the number of particles in the filter
    * @param p the parameters used to run the filter
    * @param conditionalLl conditional likelihood of the observations given a value of the state
    * @param state the conditioned upon state
    * @param mod the specification of the system evolution matrix, G and observation matrix F
    * @param obs a list of observations
    * @return a tuple containing the log-likelihood of the parameters given the observations and 
    * a single state path deterministically chosen to be the final path
    */
  def filter(
    n:   Int, 
    p:   Dlm.Parameters, 
    mod: Model, 
    obs: List[Data])(state: List[(Double, DenseVector[Double])]) = {

    val firstTime = obs.map(d => d.time).min
    val x0 = initState(p).sample(n-1).toList.map(x => (firstTime - 1, x))
    val init = State(List(x0), List.fill(n - 1)(1.0 / n), 0.0)

    val filtered = (obs, state.map(_._2)).
      zipped.
      foldLeft(init)(step(mod, p))

    sampleState(filtered.states, filtered.weights) map ((filtered.ll, _))
  }
}
