import org.scalacheck._
import Prop.forAll
import dlm.model._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Rand
import breeze.stats.covmat
import cats._
import cats.data._
import cats.implicits._
import Arbitrary.arbitrary

trait BreezeGenerators {
  val denseVector = (n: Int) => Gen.containerOfN[Array, Double](n, arbitrary[Double]).
    map(a => DenseVector(a))

  val denseMatrix = (n: Int) => Gen.containerOfN[Array, Double](n * n, arbitrary[Double]).
    map(a => new DenseMatrix(n, n, a))

  val symmetricPosDefMatrix = (n: Int) => denseVector(n).
    flatMap(z => denseVector(n).map(a => a * a.t).suchThat(m => z.t * m * z > 0))

  def dof(p: Int) = arbitrary[Double].suchThat(a => a > p - 1)
}

object WishartDistribution extends Properties("Wishart") with BreezeGenerators {
  val input = for {
    scale <- symmetricPosDefMatrix(2)
    n <- dof(scale.cols)
  } yield (n, scale)

  property("Wishart distribution should have mean nu * scale") = Prop.forAll(input) { case (nu, scale) =>
    val w = Wishart(nu, scale)
    val samples = w.sample(10000)
    w.mean === (samples.reduce(_ + _) / samples.length.toDouble)
  }
}

object MvnDistribution extends Properties("MVN") with BreezeGenerators {
  val covariance = Gen.const(DenseMatrix((0.666, -0.333), (-0.333, 0.666)))

 /**
    * Calculate the mean and covariance of a sequence of DenseVectors
    */
  def meanCovSamples(samples: Seq[DenseVector[Double]]) = {
    val n = samples.size
    val m = new DenseMatrix(n, samples.head.size, samples.map(_.data).toArray.transpose.flatten)
    val sampleMean = samples.reduce(_ + _).map(_ * 1.0/n)
    val sampleCovariance = covmat.matrixCovariance(m)

    (sampleMean, sampleCovariance)
  }
  
  // mean zero and covariance the given matrix, for some reason the off diagonals are much smaller than expected
  property("draw should sample realisations from the MVN distribution") = Prop.forAll(covariance) { cov =>

    val mvn = MultivariateGaussianSvd(DenseVector.zeros[Double](2), cov)
    val samples = mvn.sample(10000)
    
    val (sampleMean, sampleCovariance)  = meanCovSamples(samples)

    mvn.mean === sampleMean
    mvn.variance === sampleCovariance
  }
}
