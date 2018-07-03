package core.dlm.model

import breeze.stats.distributions._
import math.{sqrt, Pi}
import breeze.numerics.log
import breeze.linalg._

case class MultivariateGaussianSvd(
    mu: DenseVector[Double],
    cov: DenseMatrix[Double])(implicit rand: RandBasis = Rand)
    extends ContinuousDistr[DenseVector[Double]] {

  private val root = eigSym(cov)

  /**
    * Draw from a multivariate gaussian using eigen decomposition which
    * is often more stable than using the cholesky decomposition
    */
  def draw = {
    val x = DenseVector.rand(mu.length, rand.gaussian(0, 1))
    mu + (root.eigenvectors * diag(root.eigenvalues.mapValues(sqrt)) * x)
  }

  def mean: DenseVector[Double] = mu
  def mode: DenseVector[Double] = mu
  def variance: DenseMatrix[Double] = cov

  def logNormalizer: Double = {
    // the product of the eigenvalues is equal to the determinant of A
    // hence the sum the log of the eigenvalues is equal to the log determinant of A
    val det = sum(log(root.eigenvalues))
    mean.length * 0.5 * log(2 * Pi) + sqrt(det)
  }

  def unnormalizedLogPdf(x: DenseVector[Double]): Double = {
    val centered = x - mu
    val slv = cov \ centered

    -(slv dot centered) * 0.5
  }

}
