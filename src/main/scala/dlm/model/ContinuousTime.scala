package dlm.model

import breeze.linalg._
import breeze.stats.distributions._

object ContinuousTime {
  type TimeIncrement = Double

  /**
    * A continuous time model has a G matrix which depends on the 
    * time increment between latent-state realisations
    */
  case class Model(
    observation: (DenseVector[Double], DenseMatrix[Double]) => Rand[DenseVector[Double]],
    f: Time => DenseMatrix[Double],
    g: TimeIncrement => DenseMatrix[Double],
    conditionalLikelihood: (Dlm.Parameters) => (Observation, DenseVector[Double]) => Double
  )

  def angle(period: Int)(dt: TimeIncrement): Double = {
    2 * math.Pi * (dt % period) / period
  }

  def seasonalG(
    period:    Int,
    harmonics: Int)(
    dt:        TimeIncrement): DenseMatrix[Double] = {

    val matrices = (delta: TimeIncrement) => (1 to harmonics).
      map(h => Dlm.rotationMatrix(h * angle(period)(delta)))

    matrices(dt).reduce(Dlm.blockDiagonal)
  }

  /**
    *  Infinitessimal rotation
    *  w small, then cos(w) ~ 1, sin(w) ~ w and R = ((1, -w), (w, 1))
    *  Then for a small rotation we have (I + Adw) x
    */
  // def infinitessimalRotation(theta: Double)(x: DenseMatrix[Double]) = {

  //   val i = DenseMatrix.eye[Double](2)
  //   val r = DenseMatrix((0.0, -1.0), (1.0, 0.0))

  //   x * (i + r * theta)
  // }

  def dglm2Model(mod: Dglm.Model, contG: TimeIncrement => DenseMatrix[Double]) = {
    Model(
      mod.observation,
      (t: Time) => mod.f(t),
      contG,
      mod.conditionalLikelihood
    )
  }

  def dlm2Model(mod: Dlm.Model, contG: TimeIncrement => DenseMatrix[Double]) = {
    Model(
      (x: DenseVector[Double], v: DenseMatrix[Double]) => MultivariateGaussianSvd(x, v),
      (t: Time) => mod.f(t),
      contG,
      (p: Dlm.Parameters) => (x: DenseVector[Double], y: DenseVector[Double]) => MultivariateGaussianSvd(x, p.v).logPdf(y)
    )
  }

  def simulate(times: Iterable[Double], mod: Model, p: Dlm.Parameters) = {
    val init = (times.head, MultivariateGaussianSvd(p.m0, p.c0).draw)

    val state = times.tail.scanLeft(init) { (x, t) =>
      val dt = t - x._1
      (t, MultivariateGaussianSvd(mod.g(dt) * x._2, p.w * dt).draw)
    }

    state.map { case (t, x) => 
      (Data(t, Some(MultivariateGaussianSvd(mod.f(t).t * x, p.v).draw)), x) 
    }
  }
}
