package dlm.model

import breeze.stats.distributions._
import math.sqrt
import breeze.linalg._

case class MultivariateGaussianSvd(mu: DenseVector[Double], cov: DenseMatrix[Double])(implicit rand: RandBasis = Rand) extends Rand[DenseVector[Double]] {
  private val root = eigSym(cov)

  /**
    * Draw from a multivariate gaussian using eigen decomposition which 
    * is often more stable than using the cholesky decomposition
    */
  def draw = {
    val x = DenseVector.rand(mu.length, rand.gaussian(0, 1))
    mean + (root.eigenvectors * diag(root.eigenvalues.mapValues(sqrt)) * x)
  }

  def mean: DenseVector[Double] = mu
  def mode: DenseVector[Double] = mu
  def variance: DenseMatrix[Double] = cov
}
