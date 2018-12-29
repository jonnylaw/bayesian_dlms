package com.github.jonnylaw.dlm

import breeze.linalg.DenseVector
import breeze.stats.distributions._
import breeze.numerics.{exp, log}
import breeze.stats.mean
import cats.{Traverse, Functor}
import cats.implicits._
import scala.language.higherKinds

/**
  * State of the Bootstrap Particle Filter
  */
case class PfState(time: Double,
                   state: Vector[DenseVector[Double]],
                   weights: Vector[Double],
                   ll: Double)

/**
  * A bootstrap particle filter which can be used for inference of
  * Dynamic Generalised Linear Models (DGLMs),
  * where the observation distribution is not Gaussian.
  * @param n the number of particles used in the filter
  * @param n0 if ESS < n0 then resample
  */
case class ParticleFilter(n: Int,
                          n0: Int,
                          resample: (
                              Vector[DenseVector[Double]],
                              Vector[Double]) => Vector[DenseVector[Double]])
    extends FilterTs[PfState, DlmParameters, Dglm] {

  import ParticleFilter._

  /**
    * A single step of the Bootstrap Particle Filter
    */
  def step(mod: Dglm, p: DlmParameters)(s: PfState, d: Data): PfState = {

    val y = KalmanFilter.flattenObs(d.observation)
    val dt = d.time - s.time

    if (y.data.isEmpty) {
      val x = advanceState(dt, s.state, mod, p).draw
      s.copy(state = x)
    } else {
      val x1 = advanceState(dt, s.state, mod, p).draw
      val w = calcWeights(mod, d.time, x1, d.observation, p)
      val max = w.max
      val w1 = w map (a => exp(a - max))
      val ll = s.ll + max + log(mean(w1))
      val ess = effectiveSampleSize(normaliseWeights(w1))

      if (ess < n0) {
        val resampled = resample(x1, w1)
        PfState(d.time, resampled, Vector.fill(n)(1.0 / n), ll)
      } else {
        PfState(d.time, x1, w1, ll)
      }
    }
  }

  def initialiseState[T[_]: Traverse](model: Dglm,
                                      p: DlmParameters,
                                      ys: T[Data]): PfState = {

    val initState = MultivariateGaussian(p.m0, p.c0).sample(n).toVector
    val t0 = ys.map(_.time).reduceLeftOption((t0, d) => math.min(t0, d))
    PfState(t0.get, initState, Vector.fill(n)(1.0 / n), 0.0)
  }
}

object ParticleFilter {

  /**
    * Run a Bootstrap Particle Filter over a DGLM to calculate the log-likelihood
    */
  def likelihood[T[_]: Traverse](mod: Dglm, ys: T[Data], n: Int)(
      p: DlmParameters): Double = {

    val n0 = math.floor(n / 5).toInt
    val filter = ParticleFilter(n, n0, multinomialResample)
    val init = filter.initialiseState(mod, p, ys)
    ys.foldLeft(init)(filter.step(mod, p)).ll
  }

  /**
    * Advance the system of particles to the next timepoint
    * @param dt the time increment to the next observation
    * @param state a collection of particles representing time emprical posterior
    * distribution of the state at time - 1
    * @return a distribution over a collection of particles at this time
    */
  def advanceState[T[_]: Traverse](dt: Double,
                                   state: T[DenseVector[Double]],
                                   model: Dglm,
                                   p: DlmParameters) = {

    Rand.always(state map (x => Dglm.stepState(model, p)(x, dt).draw))
  }

  def calcWeight(mod: Dglm,
                 time: Double,
                 x: DenseVector[Double],
                 y: DenseVector[Option[Double]],
                 p: DlmParameters) = {

    val fm = KalmanFilter.missingF(mod.f, time, y)
    val vm = KalmanFilter.missingV(p.v, y)
    mod.conditionalLikelihood(vm)(fm.t * x, KalmanFilter.flattenObs(y))
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
  def calcWeights[F[_]: Functor](mod: Dglm,
                                 time: Double,
                                 state: F[DenseVector[Double]],
                                 y: DenseVector[Option[Double]],
                                 p: DlmParameters) = {

    state.map(x => calcWeight(mod, time, x, y, p))
  }

  /**
    * Multinomial Resampling
    * @param particles the particles representing the prior distribution of the state at time t
    * @param weights the conditional likelihood of each particle given the observation
    * @return a random sample from a Multinomial distribution with probabilities equal to the weights
    */
  def multinomialResample[A](particles: Vector[A],
                             weights: Vector[Double]): Vector[A] = {

    val indices =
      Multinomial(DenseVector(weights.toArray)).sample(particles.size).toVector

    indices map (particles(_))
  }

  def discreteUniform(min: Int, max: Int): Rand[Int] = {
    Rand.always(min + scala.util.Random.nextInt(max - min + 1))
  }

  /**
    * Metropolis resampling which only computes the ratio between pairs of weights making
    * it amenable to parallelisation
    * @param b the number of iterations
    * @param w a vector of normalised weights
    */
  def metropolisResampling[A](b: Int)(x: Vector[A],
                                      w: Vector[Double]): Vector[A] = {
    val n = w.size

    def loop(b: Int, k: Int): Int = {
      if (b == 0) {
        k
      } else {
        val u = Uniform(0, 1).draw
        val j = discreteUniform(0, n - 1).draw
        if (u <= w(j) / w(k)) {
          loop(b - 1, j)
        } else {
          loop(b - 1, k)
        }
      }
    }

    w.indices.toVector.par.map { i =>
      val k = loop(b, i)
      x(k)
    }.seq
  }

  /**
    * Calculate the effective sample size of a particle cloud
    * allowing
    */
  def effectiveSampleSize(ws: Seq[Double]): Int = {
    val ss = ws.map(w => w * w).sum
    math.floor(1 / ss).toInt
  }

  /**
    * Normalise a sequence of weights so they sum to one
    */
  def normaliseWeights[F[_]: Traverse](w: F[Double]): F[Double] = {
    val total = w.foldLeft(0.0)(_ + _)
    w map (_ / total)
  }

  /**
    * Perform systematic resampling
    */
  def systematicResample(w: Seq[Double]) = {
    val cumSum = w.scanLeft(0.0)(_ + _)

    val uk = w.zipWithIndex map {
      case (_, i) => scala.util.Random.nextDouble + i / w.size
    }

    uk flatMap (u => {
      cumSum.zipWithIndex
        .filter { case (wn, index) => wn < u }
        .map { case (_, i) => i - 1 }
        .take(1)
    })
  }
}
