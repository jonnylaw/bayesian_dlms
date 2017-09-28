package dlm.model

import breeze.stats.distributions._
import breeze.linalg._
import breeze.numerics._

case class InverseWishart(
  nu: Double,
  psi: DenseMatrix[Double])(implicit rand: RandBasis = Rand) extends ContinuousDistr[DenseMatrix[Double]] with Moments[DenseMatrix[Double], DenseMatrix[Double]] {

  val d: Int = psi.cols

  def logNormalizer: Double = ???

  def unnormalizedLogPdf(x: DenseMatrix[Double]): Double = ???

  def draw(): DenseMatrix[Double] = {
        val a: DenseMatrix[Double] = DenseMatrix.tabulate(d, d){ case (i, j) =>
      (for {
        c <- ChiSquared(nu - i + 1)
        n <- Gaussian(0, 1)
        x = if (i == j) sqrt(c) else if (i > j) n else 0
      } yield x).draw
    }

    val l = cholesky(inv(psi))
    val invl = inv(l)
    val inva = inv(a)
    invl.t * inva.t * inva * invl
  }

  def entropy: Double = ???
  def mean: breeze.linalg.DenseMatrix[Double] = ???
  def mode: breeze.linalg.DenseMatrix[Double] = ???
  def variance: breeze.linalg.DenseMatrix[Double] = ???
}
