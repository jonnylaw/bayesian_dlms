package dlm.core.model

// exclude vector
import breeze.linalg.{Vector => _, _}
import breeze.stats.distributions._
import cats.Traverse
import cats.implicits._
import math.{log, exp}
import LiuAndWestFilter._

case class RbState(
  time:    Double,
  params:  Vector[DlmParameters],
  mt:      Vector[DenseVector[Double]],
  ct:      Vector[DenseMatrix[Double]],
  weights: Vector[Double])

/**
  * Extended Particle filter which approximates 
  * the parameters as a particle cloud
  */
case class RaoBlackwellFilter(n: Int, prior: Rand[DlmParameters], a: Double)
    extends FilterTs[RbState, DlmParameters, Dlm] {

  def initialiseState[T[_]: Traverse](
    model: Dlm,
    p: DlmParameters,
    ys: T[Data]): RbState = {

    val t0 = ys.map(_.time).
      reduceLeftOption((t0, d) => math.min(t0, d))

    val m0 = Vector.fill(n)(p.m0)
    val c0 = Vector.fill(n)(p.c0)
    val p0 = prior.sample(n).map(_.map(log))
    val w = Vector.fill(n)(1.0 / n)
    RbState(t0.get - 1.0, p0.toVector, m0, c0, w)
  }

  /**
    * Perform the kalman filter and return the 
    * log-likelihood of a given sample
    */
  def kfStep(
    mod:    Dlm,
    dt:     Double,
    mt:     DenseVector[Double],
    ct:     DenseMatrix[Double],
    weight: Double,
    params: DlmParameters,
    y:      Data) = {

    val kf = KalmanFilter(KalmanFilter.advanceState(params, mod.g))
    val (at, rt) = KalmanFilter.advState(mod.g, mt, ct, dt, params.w map exp)
    val (_, _, mt1, ct1, w1) = kf.updateState(mod.f, at, rt, y, params.v map exp, weight)

    (mt1, ct1, w1)
  }

  def step(
    mod: Dlm,
    p:   DlmParameters)
    (x:  RbState, d: Data): RbState = {

    val varParams = varParameters(x.params)
    val mi = scaleParameters(x.params, a)

    val y = KalmanFilter.flattenObs(d.observation)
    val auxVars = auxiliaryVariables(x.weights, x.mt, mod, y, mi)

    // propose new log-parameters
    val propVariance = diag(varParams * (1 - a * a))
    val newParams = auxVars.
      map(i => proposal(mi(i), propVariance))

    // update the state
    val dt = d.time - x.time

    // use each parameter particle for a different Kalman Filter
    val (mean, covariance, ws) =  (x.mt, x.ct, x.weights, newParams).
      parMapN { case (m, c, w, ps) => kfStep(mod, dt, m, c, w, ps, d) }.unzip3

    // resample the states and parameters
    val maxWeight = ws.max
    val weights = ws map { a => exp(a - maxWeight) }
    val indices = ParticleFilter.
      multinomialResample(ws.indices.toVector, weights)

    val (rMean, rCov, rParams) = indices.
      map(i => (mean(i), covariance(i), newParams(i))).
      unzip3

    RbState(d.time, rParams, rMean, rCov, weights)
  }
}
