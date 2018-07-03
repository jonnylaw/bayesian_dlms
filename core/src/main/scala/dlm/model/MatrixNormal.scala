package core.dlm.model

import breeze.stats.distributions._
import breeze.linalg._

/**
  * A Normal distribution over matrices
  * @param mu the location of the distribution
  * @param u the variance of the rows
  * @param v the variance of the columns
  */
case class MatrixNormal(
    mu: DenseMatrix[Double],
    u: DenseMatrix[Double],
    v: DenseMatrix[Double]
)(implicit rand: RandBasis = Rand)
    extends ContinuousDistr[DenseMatrix[Double]] {

  private val rootRow = cholesky(u)
  private val rootCol = cholesky(v)

  /**
    * Draw from a matrix normal distribution using the cholesky decomposition
    * of the row and column covariance matrices
    */
  def draw = {
    val x = DenseVector.rand(mu.cols * mu.rows, rand.gaussian(0, 1))
    mu + rootRow * x * rootCol
  }

  def logNormalizer: Double = ???

  def unnormalizedLogPdf(x: DenseMatrix[Double]) = ???
}
