package dlm.model

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.distributions._
import breeze.numerics.{exp, log}
import breeze.stats.mean
import Dglm._
import cats.implicits._
import cats.{Traverse, Functor}
import scala.language.higherKinds
import Dlm.Data
/**
  * The particle filter can be used for inference of 
  * Dynamic Generalised Linear Models (DGLMs),
  * where the observation distribution is not Gaussian.
  */
object ParticleFilter {

  /**
    * State of the Particle Filter
    */
  case class State(
    time:    Double,
    state:   Vector[DenseVector[Double]],
    weights: Vector[Double],
    ll:      Double
  )

  /**
    * Advance the system of particles to the next timepoint
    * @param g the system evolution matrix, G_t
    * @param time the time of the next observation
    * @param state a collection of particles representing time emprical posterior 
    * distribution of the state at time - 1 
    * @param p the parameters of the model, containing the system evolution matrix, W
    * @return a distribution over a collection of particles at this time
    */
  def advanceState[F[_]: Traverse](
    g:     Double => DenseMatrix[Double],
    time:  Double, 
    state: F[DenseVector[Double]], 
    p:     Dlm.Parameters) = {

    state traverse (x => 
      MultivariateGaussianSvd(g(time) * x, p.w): Rand[DenseVector[Double]])
  }

  /**
    * Calculate the weights of each particle using the conditional likelihood of
    * the observations given the state
    * @param mod the DGLM model specification containing F_t and G_t, 
    * and the conditional likelihood of observations
    * @param time the time of the observation
    * @param state a collection of particles representing time emprical prior 
    * distribution of the state at this time
    * @param y the observation 
    * observation given a value of the state, p(y_t | F_t x_t)
    * @return a collection of weights corresponding to each particle
    */
  def calcWeights[F[_]: Functor](
    mod:    Model, 
    time:   Double, 
    state:  F[DenseVector[Double]], 
    y:      DenseVector[Option[Double]],
    p:      Dlm.Parameters
  ) = {
    val fm = KalmanFilter.missingF(mod.f, time, y)
    val vm = KalmanFilter.missingV(p.v, y)
    state.map(x => mod.conditionalLikelihood(vm)(fm.t * x, KalmanFilter.flattenObs(y)))
  }

  /**
    * Multinomial Resampling
    */
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
    p:      Dlm.Parameters)
    (s:     State,
     d:     Data): State = {

    val y = KalmanFilter.flattenObs(d.observation)

    if (y.data.isEmpty) {
      val x = advanceState(mod.g, d.time, s.state, p).draw
      val n = s.state.size
      State(d.time, x, Vector.fill(n)(1.0 / n), s.ll)
    } else {
      val resampledX = resample(s.state, s.weights)
      val x1 = advanceState(mod.g, d.time, resampledX, p).draw
      val w = calcWeights(mod, d.time, x1, d.observation, p)
      val max = w.max
      val w1 = w map (a => exp(a - max))
      val ll = s.ll + max + log(mean(w1))

      State(d.time, x1, w1, ll)
    }
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
    val init = State(
      observations.map(_.time).min - 1,
      initState, 
      Vector.fill(n)(1.0 / n), 0.0)

    observations.scanLeft(init)(filterStep(mod, p))
  }

  /**
    * Run a Bootstrap Particle Filter over a DGLM to calculate the log-likelihood
    */
  def likelihood(
    mod:          Model, 
    observations: Array[Data], 
    n:            Int)(
    p:            Dlm.Parameters) = {

    val initState = MultivariateGaussian(p.m0, p.c0).sample(n).toVector
    val init = State(
      observations.map(_.time).min - 1, 
      initState,
      Vector.fill(n)(1.0 / n), 0.0)

    observations.foldLeft(init)(filterStep(mod, p)).ll
  }
}
