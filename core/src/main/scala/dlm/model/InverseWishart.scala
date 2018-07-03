package core.dlm.model

import breeze.stats.distributions._
import breeze.linalg._

case class InverseWishart(nu: Double, psi: DenseMatrix[Double])(
    implicit rand: RandBasis = Rand)
    extends ContinuousDistr[DenseMatrix[Double]]
    with Moments[DenseMatrix[Double], DenseMatrix[Double]] {

  val d: Int = psi.cols

  def logNormalizer: Double = ???

  def unnormalizedLogPdf(x: DenseMatrix[Double]): Double = ???

  private val l = cholesky(inv(psi))

  def draw(): DenseMatrix[Double] = {
    val a = Wishart(nu, psi).bartlettDecomp()

    val invl = inv(l)
    val inva = inv(a)
    invl.t * inva.t * inva * invl
  }

  def entropy: Double = ???
  def mean: breeze.linalg.DenseMatrix[Double] = psi / (nu - d - 1)
  def mode: breeze.linalg.DenseMatrix[Double] = psi / (nu - d - 1)
  def variance: breeze.linalg.DenseMatrix[Double] = ???
}
