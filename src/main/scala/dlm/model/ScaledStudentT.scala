package dlm.model

import breeze.stats.distributions._
import breeze.linalg._
import breeze.numerics._

case class ScaledStudentsT(
  dof:      Double,
  location: Double,
  scale:    Double)
  (implicit rand: RandBasis = Rand) 
    extends ContinuousDistr[Double] with Moments[Double, Double] {

  def logNormalizer: Double = 
    -lgamma((dof + 1) * 0.5) + 0.5 * log(math.Pi * dof * scale) + lgamma(dof * 0.5)

  def unnormalizedLogPdf(x: Double): Double = 
    -(dof + 1) * 0.5 * log(1 + ((x - location) * (x - location)) / (dof * scale * scale))

  def draw(): Double = {
    val alpha = dof * 0.5
    val beta = dof * scale * scale * 0.5
    val res: Rand[Double] = for {
      v <- InverseGamma(alpha, beta)
      x <- Gaussian(location, sqrt(v))
    } yield x

    res.draw
  }

  def entropy: Double = ???

  /**
    * The mean is defined for dof > 1
    */
  def mean: Double = location
  def mode: Double = location

  /**
    * The variance is defined for dof > 2
    */
  def variance: Double = scale * scale * (dof / (dof - 2))
}
