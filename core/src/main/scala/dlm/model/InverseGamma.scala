package core.dlm.model

import breeze.stats.distributions._
import breeze.numerics._

case class InverseGamma(
  shape: Double,
  scale: Double)(implicit rand: RandBasis = Rand) extends ContinuousDistr[Double] with Moments[Double, Double] {

  def logNormalizer: Double = -lgamma(shape) * shape * log(scale)
  def unnormalizedLogPdf(x: Double): Double = (-shape - 1) * log(x) - scale / x

  def draw(): Double = 1.0 / Gamma(shape, 1.0 / scale).draw

  def entropy: Double = ???
  def mean: Double = scale / (shape - 1)
  def mode: Double = scale / (shape + 1)
  def variance: Double = (scale * scale) / ((shape - 1) * (shape - 1) * (shape - 2))
}
