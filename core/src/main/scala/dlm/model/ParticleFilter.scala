package core.dlm.model

import breeze.linalg.DenseVector
import breeze.stats.distributions._
import breeze.numerics.{exp, log}
import breeze.stats.mean
import cats.implicits._
import cats.{Traverse, Functor}
import scala.language.higherKinds
import Dlm.Data

/**
  * State of the Bootstrap Particle Filter
  */
case class PfState(
  time:    Double,
  state:   Vector[DenseVector[Double]],
  weights: Vector[Double],
  ll:      Double)

/**
  * The particle filter can be used for inference of
  * Dynamic Generalised Linear Models (DGLMs),
  * where the observation distribution is not Gaussian.
  */
object ParticleFilter {

  /**
    * Advance the system of particles to the next timepoint
    * @param dt the time increment to the next observation
    * @param state a collection of particles representing time emprical posterior
    * distribution of the state at time - 1
    * @param p the parameters of the model, containing the system evolution matrix, W
    * @return a distribution over a collection of particles at this time
    */
  def advanceState[F[_]: Traverse](model: DglmModel,
                                   dt: Double,
                                   state: F[DenseVector[Double]],
                                   p: DlmParameters) = {

    state traverse (x => Dglm.stepState(model, p, x, dt))
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
  def calcWeights[F[_]: Functor](mod: DglmModel,
                                 time: Double,
                                 state: F[DenseVector[Double]],
                                 y: DenseVector[Option[Double]],
                                 p: DlmParameters) = {

    val fm = KalmanFilter.missingF(mod.f, time, y)
    val vm = KalmanFilter.missingV(p.v, y)
    state.map(x =>
      mod.conditionalLikelihood(vm)(fm.t * x, KalmanFilter.flattenObs(y)))
  }

  /**
    * Multinomial Resampling
    * @param particles the particles representing the prior distribution of the state at time t
    * @param weights the conditional likelihood of each particle given the observation
    * @return a random sample from a Multinomial distribution with probabilities equal to the weights
    */
  def resample[A](particles: Vector[A], weights: Vector[Double]): Vector[A] = {

    val indices =
      Multinomial(DenseVector(weights.toArray)).sample(particles.size).toVector

    indices map (particles(_))
  }

  /**
    * A single step of the Bootstrap Particle Filter
    */
  def step(mod: DglmModel,
    p:          DlmParameters)
    (s: PfState, d: Data): PfState = {

    val y = KalmanFilter.flattenObs(d.observation)
    val dt = d.time - s.time

    if (y.data.isEmpty) {
      val x = advanceState(mod, dt, s.state, p).draw
      val n = s.state.size
      PfState(d.time, x, Vector.fill(n)(1.0 / n), s.ll)
    } else {
      val resampledX = resample(s.state, s.weights)
      val x1 = advanceState(mod, dt, resampledX, p).draw
      val w = calcWeights(mod, d.time, x1, d.observation, p)
      val max = w.max
      val w1 = w map (a => exp(a - max))
      val ll = s.ll + max + log(mean(w1))

      PfState(d.time, x1, w1, ll)
    }
  }

  def initialiseState[T[_]: Traverse](
    model: DglmModel,
    p: DlmParameters,
    ys: T[Data],
    n: Int): PfState = {

    val initState = MultivariateGaussian(p.m0, p.c0).sample(n).toVector
    PfState(ys.foldLeft(0.0)((t0, d) => math.min(t0, d.time)),
            initState,
            Vector.fill(n)(1.0 / n),
            0.0)
  }

  /**
    * Perform the Filter on a traversable collection
    */
  def filter[T[_]: Traverse](model: DglmModel, ys: T[Data], p: DlmParameters, n: Int): T[PfState] = {

    val init = initialiseState(model, p, ys, n)
    Filter.scanLeft(ys, init, step(model, p))
  }

  /**
    * Run a Bootstrap Particle Filter over a DGLM to calculate the log-likelihood
    */
  def likelihood[T[_]: Traverse](
    mod: DglmModel,
    ys: T[Data],
    n: Int)
    (p: DlmParameters) = {

    val init = initialiseState(mod, p, ys, n)
    ys.foldLeft(init)(step(mod, p)).ll
  }
}
