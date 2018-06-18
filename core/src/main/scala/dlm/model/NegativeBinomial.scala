package core.dlm.model

import breeze.stats.distributions._
import breeze.numerics._

case class NegativeBinomial(
  location: Double,
  scale:    Double)
  (implicit rand: RandBasis = Rand) extends DiscreteDistr[Int] with Moments[Double, Double]  {

  def logNormalizer: Double = ???

  val p = location / (scale + location)

  override def logProbabilityOf(k: Int) = {
    lgamma(scale + k) - lgamma(k + 1) - lgamma(scale) + scale * math.log(1 - p) + k * math.log(p)
  }

  def probabilityOf(x: Int): Double = exp(logProbabilityOf(x))

  def draw(): Int = {
    val res = for {
      lambda <- Gamma(scale, p / (1-p))
      x <- Poisson(lambda)
    } yield x

    res.draw
  }

  def mean = p * scale / (1 - p)

  def variance: Double = p * scale / math.pow(1 - p, 2)

  def entropy: Double = ???
  def mode: Double = ???
}


