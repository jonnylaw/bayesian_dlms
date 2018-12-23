package dlm.core.model

import breeze.stats.distributions._
import breeze.linalg._
import breeze.numerics._

case class MultivariateStudentsT(
    dof: Double,
    location: DenseVector[Double],
    shape: DenseMatrix[Double])(implicit rand: RandBasis = Rand)
    extends ContinuousDistr[DenseVector[Double]]
    with Moments[DenseVector[Double], DenseMatrix[Double]] {

  def logNormalizer: Double = ???

  def unnormalizedLogPdf(x: DenseVector[Double]): Double = ???

  def draw(): DenseVector[Double] = {
    val out = for {
      z <- MultivariateGaussianSvd(DenseVector.zeros[Double](location.size),
                                   shape)
      u <- Gamma(dof * 0.5, 2)
      x = z /:/ sqrt(u / dof) + location
    } yield x

    out.draw
  }

  def entropy: Double = ???

  /**
    * The mean is defined for dof > 1
    */
  def mean: DenseVector[Double] = location
  def mode: DenseVector[Double] = location

  /**
    * The variance is defined for dof > 2
    */
  def variance: DenseMatrix[Double] = shape * (dof / (dof - 2))
}
