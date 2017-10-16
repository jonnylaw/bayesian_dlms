package dlm.model

import breeze.stats.distributions._
import breeze.linalg._
import breeze.numerics._

case class Wishart(
  n: Double,
  scale: DenseMatrix[Double])(implicit rand: RandBasis = Rand) extends ContinuousDistr[DenseMatrix[Double]] with Moments[DenseMatrix[Double], DenseMatrix[Double]] {

  val d: Int = scale.cols

  def lgmultivariategamma(x: Double): Double = {
    log(math.Pi) * d * (d - 1) * 0.25 + ((1 to d) map (i => lgamma(x + (1 - i) * 0.5))).sum
  }

  def logNormalizer: Double = {
    -logdet(scale)._2 * 0.5 * n - n * d * 0.5 * log(2) - lgmultivariategamma(n * 0.5)
  }

  def unnormalizedLogPdf(x: DenseMatrix[Double]): Double = {
    logdet(x)._2 * (n - d - 1) * 0.5 - 0.5 * trace(scale \ x)
  }

  private val l = cholesky(scale)

  def draw(): DenseMatrix[Double] = drawBartlett()

  // Bartlett Decomposition
  // https://en.wikipedia.org/wiki/Wishart_distribution#Bartlett_decomposition
  def bartlettDecomp() = {
    DenseMatrix.tabulate(d, d){ case (i, j) =>
      (for {
        c <- ChiSquared(n - i)
        n <- Gaussian(0, 1)
        x = if (i == j) sqrt(c) else if (i > j) n else 0
      } yield x).draw
    }
  }

  def drawBartlett() = {
    val a = bartlettDecomp()

    l * a * a.t * l.t
  }

  /**
    *  Draw from the wishart using the definition of the Wishart distribution
    */
  def drawNaive(): DenseMatrix[Double] = {
    val xi = Vector.fill(n.toInt)(l * DenseVector.rand(scale.cols, rand.gaussian(0, 1)))
    xi.map(x => x * x.t).reduce(_ + _)
  }

  def entropy: Double = ???
  def mean: breeze.linalg.DenseMatrix[Double] = n * scale
  def mode: breeze.linalg.DenseMatrix[Double] = (n - d - 1) * scale
  def variance: breeze.linalg.DenseMatrix[Double] = ???
}
