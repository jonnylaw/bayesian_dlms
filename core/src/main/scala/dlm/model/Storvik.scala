package dlm.core.model

import breeze.linalg.{DenseVector, diag}
import breeze.stats.distributions._
import breeze.numerics.exp
import cats.implicits._
import cats.Traverse
import scala.language.higherKinds
import spire.syntax.cfor._

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
  statsV: Vector[Vector[InverseGamma]],
  statsW: Vector[Vector[InverseGamma]],
  ess:    Int)

object StorvikFilter {

  def initialiseState[T[_]: Traverse](
    model: Dglm,
    p: DlmParameters,
    ys: T[Data],
    n: Int,
    priorV: InverseGamma,
    priorW: InverseGamma): StorvikState = {

    val x0 = MultivariateGaussian(p.m0, p.c0).sample(n).toVector
    val vStats = Vector.fill(n)(Vector.fill(p.w.cols)(priorV))
    val wStats = Vector.fill(n)(Vector.fill(p.w.cols)(priorW))
    val p0 = drawParams(vStats, wStats, p)

    StorvikState(0.0, x0, p0, vStats, wStats, 0)
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
    mod:   Dglm,
    stats: Vector[InverseGamma],
    x0:    DenseVector[Double],
    x1:    DenseVector[Double]): Vector[InverseGamma] = {

    val diff = (x1 - mod.g(dt) * x0)
    val ss = (diff *:* diff) / dt
    val shapes = stats.map(_.shape + 0.5)
    val scales = (stats zip ss.data) map { case (s, si) => s.scale + 0.5 * si }

    (shapes zip scales) map { case (shape, scale) => InverseGamma(shape, scale) }
  }

  def drawParams(
    statsV: Vector[Vector[InverseGamma]],
    statsW: Vector[Vector[InverseGamma]],
    p: DlmParameters) = {

    (statsV, statsW).
      zipped.
      map { case (vs, ws) => p.copy(
             v = diag(DenseVector(vs.map(g => g.draw).toArray)),
             w = diag(DenseVector(ws.map(g => g.draw).toArray))
           )}
  }

  /**
    * Update the value of the sufficient statistics for the observation covariance matrix
    */
  def updateStatsV(
    time:  Double,
    mod:   Dglm,
    stats: Vector[InverseGamma],
    x: DenseVector[Double],
    y:   DenseVector[Double]): Vector[InverseGamma] = {

    val diff = (y - mod.f(time).t * x)
    val ss = (diff *:* diff)
    val shapes = stats.map(_.shape + 0.5)
    val scales = (stats zip ss.data) map { case (s, si) => s.scale + 0.5 * si }

    (shapes zip scales) map { case (shape, scale) => InverseGamma(shape, scale) }
  }

  def advanceState(
    mod: Dglm,
    dt: Double,
    ps: Vector[DlmParameters],
    xs: Vector[DenseVector[Double]]) = {

    (xs zip ps).
      map { case (x, p) => Dglm.stepState(mod, p, x, dt).draw }
  }

  def step(mod: Dglm, n0: Int)(s: StorvikState, d: Data): StorvikState = {

    println(s"Iteration ${d.time}")
    val y = KalmanFilter.flattenObs(d.observation)
    val dt = d.time - s.time

    // propose parameters from a known distribution, using the vector of sufficient statistics
    val params = drawParams(s.statsV, s.statsW, s.params.head)

    // advance the state particles, according to p(x(t) | x(t-1), psi)
    val x1 = advanceState(mod, dt, params, s.state)

    // calculate the new weights, evaluating p(y(t) | x(t), psi)
    val weights = calcWeights(mod, d.time, x1, d.observation, params)

    // avoid underflow
    val maxWeight = weights.max
    val expWeight = weights map (a => exp(a - maxWeight))

    // normalise the weights to sum to one
    val normWeights = ParticleFilter.normaliseWeights(expWeight)

    // calculate the effective sample size
    val ess = ParticleFilter.effectiveSampleSize(normWeights)

    // resample if the effective sample size is less than n0
    if (ess < n0) {
      println("Resampling")

      val indices = ParticleFilter.multinomialResample(expWeight.indices.toVector, expWeight)
      val resampledState = indices map (x1(_))
      val resampledParams = indices map (params(_))
      val resampledStatsW = indices map (s.statsW(_))
      val resampledStatsV = indices map (s.statsV(_))

      val statsW = (s.state, resampledState, resampledStatsW).
        zipped.
        map { case (x0, x, stats) => updateStatsW(dt, mod, stats, x0, x) }

      val statsV = (resampledState, resampledStatsV).
        zipped.
        map { case (x, stats) => updateStatsV(d.time, mod, stats, x, y) }

      StorvikState(d.time, resampledState, resampledParams, statsV, statsW, ess)
    } else {
      val statsW = (s.state, x1, s.statsW).
        zipped.
        map { case (x0, x, stats) => updateStatsW(dt, mod, stats, x0, x) }

      val statsV = (x1, s.statsV).
        zipped.
        map { case (x, stats) => updateStatsV(d.time, mod, stats, x, y) }

      StorvikState(d.time, x1, params, statsV, statsW, ess)
    }
  }

  def filterArrayTs(
    model: Dglm,
    ys: Vector[Data],
    p: DlmParameters,
    priorV: InverseGamma,
    priorW: InverseGamma,
    n: Int,
    n0: Int) = {

    val st = Array.ofDim[StorvikState](ys.length + 1)
    st(0) = initialiseState(model, p, ys, n, priorV, priorW)

    cfor(1)(_ < st.size, _ + 1) { i =>
      st(i) = step(model, n0)(st(i - 1), ys(i - 1))
    }

    st
  }

  def filterTs(
    model: Dglm,
    ys: Vector[Data],
    p: DlmParameters,
    priorV: InverseGamma,
    priorW: InverseGamma,
    n: Int,
    n0: Int): Vector[StorvikState] = {

    val init = initialiseState(model, p, ys, n, priorV, priorW)
    ys.scanLeft(init)(step(model, n0))
  }

  def calcWeights(
    mod: Dglm,
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
