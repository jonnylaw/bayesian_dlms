package dlm.core.model

import breeze.linalg.{DenseVector, diag}
import breeze.stats.distributions._
import breeze.numerics.exp
import cats.implicits._
import cats.Traverse
import scala.language.higherKinds
import ParticleFilter._

/**
  * State of the Storvik filter
  * @param time the time of the observation associated with this latent state
  * @param state the particle cloud representing the posterior state
  * @param params 
  */
case class StorvikState(
  time:   Double,
  state:  Vector[DenseVector[Double]],
  params: Vector[DlmParameters],
  statsW: Vector[Vector[InverseGamma]],
  statsV: Vector[Vector[InverseGamma]],
  ess:    Int)

case class StorvikFilter(
  n:      Int,
  priorW: InverseGamma,
  priorV: InverseGamma) extends FilterTs[StorvikState, DlmParameters, DglmModel] {

  def initialiseState[T[_]: Traverse](
    model: DglmModel,
    p: DlmParameters,
    ys: T[Data]): StorvikState = {

    val x0 = MultivariateGaussian(p.m0, p.c0).sample(n).toVector
    val p0 = Vector.fill(n)(p)
    val wStats = Vector.fill(n)(Vector.fill(p.w.cols)(priorW))
    val vStats = Vector.fill(n)(Vector.fill(p.w.cols)(priorV))

    StorvikState(0.0, x0, p0, wStats, vStats, 0)
  }

  /**
    * Update the value of the sufficient statistics for the system covariance matrix
    * @param dt the time difference between observations
    * @param stats the previous value of the sufficient statistics
    * @param x0 the previous value of the state particles
    * @param x1 the current value of the state particles
    * @return a vector representing the diagonal entries of the system covariance matrix
    */
  def updateStatsW(
    dt:    Double,
    mod:   DglmModel,
    stats: Vector[InverseGamma],
    x0:    DenseVector[Double],
    x1:    DenseVector[Double]): Vector[InverseGamma] = {

    val ss = (x1 - mod.g(dt) * x0) *:* (x1 - mod.g(dt) * x0) / dt
    val shapes = stats.map(_.shape + 0.5)
    val scales = (stats zip ss.data) map { case (s, si) => s.scale + 0.5 * si }

    (shapes zip scales) map { case (shape, scale) => InverseGamma(shape, scale) }
  }

  /**
    * Update the value of the sufficient statistics for the observation covariance matrix
    */
  def updateStatsV(
    time:  Double,
    mod:   DglmModel,
    stats: Vector[InverseGamma],
    x: DenseVector[Double],
    y:   DenseVector[Double]): Vector[InverseGamma] = {

    val ss = (y - mod.f(time) * x) * (y - mod.f(time) * x)
    val shapes = stats.map(_.shape + 0.5)
    val scales = (stats zip ss.data) map { case (s, si) => s.scale + 0.5 * si }

    (shapes zip scales) map { case (shape, scale) => InverseGamma(shape, scale) }
  }

  def advanceState(
    mod: DglmModel,
    dt: Double,
    ps: Vector[DlmParameters],
    xs: Vector[DenseVector[Double]]) = {

    (xs zip ps).
      map { case (x, p) => Dglm.stepState(mod, p, x, dt).draw }
  }

  def step(
    mod: DglmModel,
    p:   DlmParameters)
    (s:  StorvikState, d: Data): StorvikState = {

    val y = KalmanFilter.flattenObs(d.observation)
    val dt = d.time - s.time

    // propose parameters from a known distribution, using the vector of sufficient statistics
    val params = (s.statsV, s.statsW).
      zipped.
      map { case (vs, ws) => p.copy(
        v = diag(DenseVector(vs.map(g => g.draw).toArray)),
        w = diag(DenseVector(ws.map(g => g.draw).toArray))
      )}

    // advance the state particles, according to p(x(t) | x(t-1), psi)
    val x1 = advanceState(mod, dt, params, s.state)

    // calculate the new weights, evaluating p(y(t) | x(t), psi)
    val weights = calcWeights(mod, d.time, x1, d.observation, params)

    // avoid underflow
    val maxWeight = weights.max
    val expWeight = weights map (a => exp(a - maxWeight))

    // normalise the weights to sum to one
    val normWeights = normaliseWeights(expWeight)

    // calculate the effective sample size, could use this to decide whether to resample
    val ess = effectiveSampleSize(normWeights)

    // resample each time
    val indices = multinomialResample(expWeight.indices.toVector, expWeight)
    val resampledState = indices map (x1(_))
    val resampledParams = indices map (params(_))
    val resampledStatsW = indices map (s.statsW(_))
    val resampledStatsV = indices map (s.statsV(_))

    val statsW = (s.state, resampledState, resampledStatsW).
      zipped.
      map { case (x, x1, stats) => updateStatsW(dt, mod, stats, x, x1) }

    val statsV = (resampledState, resampledStatsV).
      zipped.
      map { case (x, stats) => updateStatsV(d.time, mod, stats, x, y) }

    StorvikState(d.time, resampledState, resampledParams, statsW, statsV, ess)
  }


  def calcWeights(
    mod: DglmModel,
    time: Double,
    state: Vector[DenseVector[Double]],
    y: DenseVector[Option[Double]],
    ps: Vector[DlmParameters]) = {

    val fm = KalmanFilter.missingF(mod.f, time, y)

    (state zip ps).map { case (x, p) =>
      val vm = KalmanFilter.missingV(p.v, y)
      mod.conditionalLikelihood(vm)(fm.t * x, KalmanFilter.flattenObs(y))
    }
  }
}
