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
  * Extended Particle filter which approximates the parameters as a particle cloud
  */
case class RaoBlackwellFilter(n: Int, prior: Rand[DlmParameters], a: Double)
    extends FilterTs[RbState, DlmParameters, Dlm] {

  def initialiseState[T[_]: Traverse](
    model: Dlm,
    p: DlmParameters,
    ys: T[Data]): RbState = {

    val t0 = ys.map(_.time).reduceLeftOption((t0, d) => math.min(t0, d))
    val m0 = Vector.fill(n)(p.m0)
    val c0 = Vector.fill(n)(p.c0)
    val p0 = prior.sample(n).map(_.map(log))
    val w = Vector.fill(n)(1.0 / n)
    RbState(t0.get - 1.0, p0.toVector, m0, c0, w)
  }

  def step(
    mod: Dlm,
    p:   DlmParameters)
    (x:  RbState, d: Data): RbState = {

    val varParams = varParameters(x.params)
    val mi = scaleParameters(x.params, a)
    val kf = KalmanFilter(KalmanFilter.advanceState(p, mod.g))

    val y = KalmanFilter.flattenObs(d.observation)
    val auxVars = auxiliaryVariables(x.weights, x.mt, mod, y, mi)

    // propose new log-parameters
    val newParams = for {
      i <- auxVars
      param = mi(i)
      p = proposal(param, diag(varParams * (1 - a * a)))
    } yield p

    // update the state
    val dt = d.time - x.time

    // use each parameter particle for a different Kalman Filter
    val (mean, covariance, ws) = (for {
      i <- x.mt.indices
      (at, rt) = KalmanFilter.advState(mod.g, x.mt(i), x.ct(i), dt, newParams(i).w map exp)
      (ft, qt, mt1, ct1, w1) = kf.updateState(mod.f, at, rt, d, newParams(i).v map exp, x.weights(i))
    } yield (mt1, ct1, w1)).unzip3

    // resample the states and parameters
    val maxWeight = ws.max
    val weights = ws map { a => exp(a - maxWeight) }
    val indices = ParticleFilter.multinomialResample(ws.indices.toVector, weights.toVector)

    val (rMean, rCov, rParams) = (for {
      i <- indices
      mt = mean.toVector(i)
      ct = covariance.toVector(i)
      psi = newParams.toVector(i)
    } yield (mt, ct, psi)).unzip3

    RbState(d.time, rParams.toVector, rMean.toVector, rCov.toVector, ws.toVector)
  }
}
