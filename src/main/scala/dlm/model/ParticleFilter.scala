package dlm.model

import breeze.linalg.DenseVector
import breeze.stats.distributions._
import breeze.numerics.{exp, log}
import breeze.stats.mean
import Dlm._
import cats.implicits._
import cats.Traverse

/**
  * The particle filter can be used for inference of DGLMs
  * ie. Dynamic Generalised Linear Models, where the observation distribution is not Gaussian
  */
object ParticleFilter {
  type CondLikelihood = (Observation, DenseVector[Double]) => Double

  /**
    * State of the Particle Filter
    */
  case class State(
    time:    Time,
    state:   Vector[DenseVector[Double]],
    weights: Vector[Double],
    ll:      Double
  )

  def advanceState[F[_]: Traverse](
    mod:   Model, 
    time:  Time, 
    state: F[DenseVector[Double]], 
    p:     Parameters) = {

    state traverse (x => MultivariateGaussianSvd(mod.g(time) * x , p.w): Rand[DenseVector[Double]])
  }

  def calcWeights[F[_]: Traverse](
    mod:    Model, 
    time:   Time, 
    state:  F[DenseVector[Double]], 
    p:      Parameters, 
    y:      Observation,
    condLl: CondLikelihood
  ) = {
    state.map(x => condLl(y, mod.f(time).t * x))
  }

  def resample[A](
    particles: Vector[A], 
    weights:   Vector[Double]): Vector[A] = {

    val indices = Vector.fill(particles.size)(Multinomial(DenseVector(weights.toArray)).draw)

    indices map (particles(_))
  }

  def filterStep(
    mod:    Model, 
    p:      Parameters,
    condLl: CondLikelihood)(state: State, d: Data): State = d.observation match {
    case Some(y) =>
      val resampledX = resample(state.state, state.weights)
      val x1 = advanceState(mod, d.time, resampledX, p).draw
      val w = calcWeights(mod, d.time, x1, p, y, condLl)
      val max = w.max
      val w1 = w map (a => exp(a - max))
      val ll = state.ll + max + log(mean(w1))

      State(d.time, x1, w1, ll)
    case None =>
      val x = advanceState(mod, d.time, state.state, p).draw
      val n = state.state.size
      State(d.time, x, Vector.fill(n)(1.0 / n), state.ll)
  }

  /**
    * Run a Boostrap Particle Filter over a DGLM
    * @param mod a Model class containing the specification of G and F
    * @param observations an array of Data
    * @param p the parameters used to run the particle filter
    * @param n the number of particles to use in the filter, more results in higher accuracy
    * @param conditionalLikelihood a function to calculate the conditional log-likelihood of an observation given
    * a value of the state, this is typically from the exponential family and contains 
    */
  def filter(
    mod:          Model, 
    observations: Array[Data], 
    p:            Parameters, 
    n:            Int, 
    condLl:       CondLikelihood) = {

    val initState = MultivariateGaussian(p.m0, p.c0).sample(n).toVector
    val init = State(observations.head.time, initState, Vector.fill(n)(1.0 / n), 0.0)

    observations.scanLeft(init)(filterStep(mod, p, condLl))
  }

  def likelihood(
    mod:          Model, 
    observations: Array[Data], 
    p:            Parameters, 
    n:            Int, 
    condLl:       CondLikelihood) = {

    val initState = MultivariateGaussian(p.m0, p.c0).sample(n).toVector
    val init = State(observations.head.time, initState, Vector.fill(n)(1.0 / n), 0.0)

    observations.foldLeft(init)(filterStep(mod, p, condLl)).ll
  }
}
