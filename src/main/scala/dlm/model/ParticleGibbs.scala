package dlm.model

import Dlm._
import breeze.linalg.DenseVector
import cats.implicits._
import math.{exp, log}
import breeze.stats.mean

/**
  * Particle Gibbs Sampler for A Dynamic Generalised Linear Model
  */
object ParticleGibbs {
  type ConditionalLl = (Observation, DenseVector[Double]) => Double

  case class State(
    states: List[List[(Time, DenseVector[Double])]],
    weights: List[Double],
    ll: Double)

  def initState(p: Parameters) = {
    MultivariateGaussianSvd(p.m0, p.c0)
  }

  def step(
    conditionalLikelihood: ConditionalLl, 
    mod: Model, 
    p: Parameters)(s: State, a: (Data, DenseVector[Double])): State = a match {

    case (Data(t, Some(y)), conditionedState) =>
      // resample using the previous weights
      val resampled = ParticleFilter.resample(s.states.head.toVector, s.weights.toVector).toList

      // advance n-1 resampled states from time t
      val x1 = ParticleFilter.advanceState(mod, t, resampled.map(_._2), p).draw

      // concat conditioned state and advanced state
      val x: List[DenseVector[Double]] = (conditionedState :: x1)

      // calculate weights of all n states at time t
      val w = ParticleFilter.calcWeights(mod, t, x, p, y, conditionalLikelihood)

      // log-sum-exp and calculate log-likelihood
      val max = w.max
      val w1 = w map (a => exp(a - max))
      val ll = s.ll + max + log(mean(w1))

      State(x1.map(x => (t, x)) :: s.states, w1, ll)

    case (Data(t, None), conditionedState) =>
      // resample using the previous weights
      val resampledX = ParticleFilter.resample(s.states.head.toVector, s.weights.toVector)

      // advance n-1 states from time t, located at the head of the list
      val x1 = ParticleFilter.advanceState(mod, t, s.states.head.map(_._2), p).draw

      State(x1.map(x => (t, x)) :: s.states, List.fill(x1.size - 1)(1.0 / x1.size), s.ll)
  }

  /**
    * Using the weights at time T (the end of all observations) sample a path from the collection
    * of paths
    * @param states a collection of paths with ancestory
    * @param weights particle weights at time T
    * @return a single path
    */
  def sampleState(states: List[List[(Time, DenseVector[Double])]], weights: List[Double]) = {
    ParticleFilter.resample(states.transpose.toVector, weights.toVector).toList.head
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
    * a single sample of the latent state
    */
  def filter(
    n:             Int, 
    p:             Parameters, 
    conditionalLl: ConditionalLl, 
    state:         List[(Time, DenseVector[Double])], 
    mod:           Model, 
    obs:           List[Data]): (Double, List[(Time, DenseVector[Double])]) = {

    val firstTime = obs.map(d => d.time).min
    val x0 = initState(p).sample(n-1).toList.map(x => (firstTime - 1, x))
    val init = State(List(x0), List.fill(n)(1.0 / n), 0.0)

    val x = (obs, state.map(_._2)).zipped.foldLeft(init)(step(conditionalLl, mod, p))

    (x.ll, sampleState(x.states, x.weights))
  }
}

/**
  * Particle Gibbs with Ancestor resampling
  */
object ParticleGibbsAncestor {

}
