package dlm.core.model

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
case class PfState(
  time:    Double,
  state:   Vector[DenseVector[Double]],
  weights: Vector[Double],
  ll:      Double)

/**
  * A bootstrap particle filter which can be used for inference of
  * Dynamic Generalised Linear Models (DGLMs),
  * where the observation distribution is not Gaussian.
  */
case class ParticleFilter(
  n: Int,
  resample: (Vector[DenseVector[Double]], Vector[Double]) => Vector[DenseVector[Double]])
    extends FilterTs[PfState, DlmParameters, DglmModel] {

  import ParticleFilter._

  /**
    * A single step of the Bootstrap Particle Filter
    */
  def step(
    mod:      DglmModel,
    p:        DlmParameters)
    (s: PfState, d: Data): PfState = {

    val y = KalmanFilter.flattenObs(d.observation)
    val dt = d.time - s.time

    if (y.data.isEmpty) {
      val x = advanceState(mod, dt, s.state, p).draw
      s.copy(state = x)
    } else {
      val x1 = advanceState(mod, dt, s.state, p).draw
      val w = calcWeights(mod, d.time, x1, d.observation, p)
      val max = w.max
      val w1 = w map (a => exp(a - max))
      val ll = s.ll + max + log(mean(w1))

      val resampled = resample(x1, w1)

      PfState(d.time, resampled, w1, ll)
    }
  }

  def initialiseState[T[_]: Traverse](
    model: DglmModel,
    p: DlmParameters,
    ys: T[Data]): PfState = {

    val initState = MultivariateGaussian(p.m0, p.c0).sample(n).toVector
    val t0 = ys.map(_.time).reduceLeftOption((t0, d) => math.min(t0, d))
    PfState(t0.get - 1.0, initState, Vector.fill(n)(1.0 / n), 0.0)
  }
}

object ParticleFilter {
  /**
    * Run a Bootstrap Particle Filter over a DGLM to calculate the log-likelihood
    */
  def likelihood[T[_]: Traverse](
    mod: DglmModel,
    ys: T[Data],
    n: Int)
    (p: DlmParameters): Double = {

    val filter = ParticleFilter(n, multinomialResample)
    val init = filter.initialiseState(mod, p, ys)
    ys.foldLeft(init)(filter.step(mod, p)).ll
  }

  /**
    * Advance the system of particles to the next timepoint
    * @param dt the time increment to the next observation
    * @param state a collection of particles representing time emprical posterior
    * distribution of the state at time - 1
    * @param p the parameters of the model, containing the system evolution matrix, W
    * @return a distribution over a collection of particles at this time
    */
  def advanceState[T[_]: Traverse](
    model: DglmModel,
    dt:    Double,
    state: T[DenseVector[Double]],
    p:     DlmParameters) = {

    state traverse (x => Dglm.stepState(model, p, x, dt))
  }

  def calcWeight(
    mod: DglmModel,
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
  def calcWeights[F[_]: Functor](
    mod:   DglmModel,
    time:  Double,
    state: F[DenseVector[Double]],
    y:     DenseVector[Option[Double]],
    p:     DlmParameters) = {

    state.map(x => calcWeight(mod, time, x, y, p))
  }

  /**
    * Multinomial Resampling
    * @param particles the particles representing the prior distribution of the state at time t
    * @param weights the conditional likelihood of each particle given the observation
    * @return a random sample from a Multinomial distribution with probabilities equal to the weights
    */
  def multinomialResample[A](particles: Vector[A], weights: Vector[Double]): Vector[A] = {

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
    * @param w
    */
  def metropolisResampling[A](b: Int)(x: Vector[A], w: Vector[Double]): Vector[A] = {
    val n = w.size

    def loop(b: Int, k: Int): Int = {
      if (b == 0) {
        k
      } else {
        val u = Uniform(0, 1).draw
        val j = discreteUniform(0, n-1).draw
        if (u <= w(j)/w(k)) {
          loop(b-1, j)
        } else {
          loop(b-1, k)
        }
      }
    }

    w.indices.toVector.par.map { i =>
      val k = loop(b, i)
      x(k)
    }.seq
  }


  def effectiveSampleSize(w: Seq[Double]): Int = {
    val ss = (w zip w).map { case (a, b) => a * b }.sum
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

    val uk = w.zipWithIndex map { case (_, i) => scala.util.Random.nextDouble + i / w.size }

    uk flatMap (u => {
      cumSum.
        zipWithIndex.
        filter { case (wn, index) => wn < u }.
        map { case (_, i) => i - 1}.
        take(1)
    })
  }
}
