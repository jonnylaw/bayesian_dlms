package dlm.model

import breeze.linalg.DenseVector
import breeze.stats.distributions._
import breeze.numerics.{exp, log}
import breeze.stats.mean
import Dglm._
import cats.implicits._
import cats.{Traverse, Functor}
import scala.language.higherKinds

/**
  * The particle filter can be used for inference of DGLMs
  * ie. Dynamic Generalised Linear Models, where the observation distribution is not Gaussian
  */
object ParticleFilter {

  /**
    * State of the Particle Filter
    */
  case class State(
    time:    Time,
    state:   Vector[DenseVector[Double]],
    weights: Vector[Double],
    ll:      Double
  )

  /**
    * Advance the system of particles to the next timepoint
    * @param mod the DLM model specification containing F_t and G_t, the observation
    * and system evolution matrices
    * @param time the time of the next observation
    * @param state a collection of particles representing time emprical posterior 
    * distribution of the state at time - 1 
    * @param p the parameters of the model, containing the system evolution matrix, W
    * @return a distribution over a collection of particles at this time
    */
  def advanceState[F[_]: Traverse](
    mod:   Model, 
    time:  Time, 
    state: F[DenseVector[Double]], 
    p:     Dlm.Parameters) = {

    state traverse (x => MultivariateGaussianSvd(mod.g(time) * x , p.w): Rand[DenseVector[Double]])
  }

  /**
    * Calculate the weights of each particle using the conditional likelihood of
    * the observations given the state
    * @param mod the DLM model specification containing F_t and G_t, the observation
    * and system evolution matrices
    * @param time the time of the observation
    * @param state a collection of particles representing time emprical prior 
    * distribution of the state at this time
    * @param y the observation 
    * observation given a value of the state, p(y_t | F_t x_t)
    * @return a collection of weights corresponding to each particle
    */
  def calcWeights[F[_]: Functor](
    mod:    Model, 
    time:   Time, 
    state:  F[DenseVector[Double]], 
    y:      Observation,
    p:      Dlm.Parameters
  ) = {
    state.map(x => mod.conditionalLikelihood(p)(mod.f(time).t * x, y))
  }

  def resample[A](
    particles: Vector[A], 
    weights:   Vector[Double]): Vector[A] = {

    val indices = Multinomial(DenseVector(weights.toArray)).
      sample(particles.size).
      toVector

    indices map (particles(_))
  }

  def filterStep(
    mod:    Model, 
    p:      Dlm.Parameters)(state: State, d: Data): State = d.observation match {
    case Some(y) =>
      val resampledX = resample(state.state, state.weights)
      val x1 = advanceState(mod, d.time, resampledX, p).draw
      val w = calcWeights(mod, d.time, x1, y, p)
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
    * @param observations a Seq of Data
    * @param p the parameters used to run the particle filter
    * @param n the number of particles to use in the filter, more results in higher accuracy
    * @param conditionalLikelihood a function to calculate the conditional log-likelihood of an observation given
    * a value of the state, this is typically from the exponential family and contains 
    */
  def filter(
    mod:          Model, 
    observations: Seq[Data], 
    p:            Dlm.Parameters, 
    n:            Int) = {

    val initState = MultivariateGaussian(p.m0, p.c0).sample(n).toVector
    val init = State(observations.head.time, initState, Vector.fill(n)(1.0 / n), 0.0)

    observations.scanLeft(init)(filterStep(mod, p))
  }

  /**
    * Run a Bootstrap Particle Filter over a DGLM to calculate the 
    */
  def likelihood(
    mod:          Model, 
    observations: Array[Data], 
    p:            Dlm.Parameters, 
    n:            Int) = {

    val initState = MultivariateGaussian(p.m0, p.c0).sample(n).toVector
    val init = State(observations.head.time, initState, Vector.fill(n)(1.0 / n), 0.0)

    observations.foldLeft(init)(filterStep(mod, p)).ll
  }
}
