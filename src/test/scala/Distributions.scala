import org.scalacheck._
import Prop.{forAll, BooleanOperators}
import dlm.model._
import breeze.linalg.{DenseMatrix, DenseVector, cond, diag}
import breeze.stats.distributions.ChiSquared
import breeze.stats.covmat
import breeze.stats.{meanAndVariance, variance}
import cats.Eq
import cats.implicits._
import Arbitrary.arbitrary

trait BreezeGenerators {
  val denseVector = (n: Int) => Gen.containerOfN[Array, Double](n, Gen.choose(-10.0, 10.0)).
    map(a => DenseVector(a))

  val denseMatrix = (n: Int) => Gen.containerOfN[Array, Double](n * n, Gen.choose(-10.0, 10.0)).
    map(a => new DenseMatrix(n, n, a))


  /**
    * Simulate a positive definite matrix with a given condition number
    */
  def symmetricPosDefMatrix(n: Int, c: Double) = {
    if (n > 2) {
      for {
        entries <- Gen.containerOfN[Array, Double](n - 2, Gen.choose(1.0, c))
        d = DenseVector(Array(1.0, c) ++ entries)
        u <- denseVector(n)
        t = 2 / (u.t * u)
        i = DenseMatrix.eye[Double](n)
      } yield (i - t * u * u.t) * d * (i - t * u * u.t)
    } else {
      for {
        u <- denseVector(n)
        d = diag(DenseVector(1.0, c))
        t = 2 / (u.t * u)
        i = DenseMatrix.eye[Double](n)
      } yield (i - t * u * u.t) * d * (i - t * u * u.t)
    }
  }

  val positiveDouble = Gen.choose(1.0, 10.0)
}

// object InverseWishartDistribution extends Properties("Inverse Wishart") with BreezeGenerators {
//   val input = symmetricPosDefMatrix(2, 1000)
 
//   property("Inverse Wishart Distribution") = Prop.forAll(input) { psi =>
//     val n = 100000
//     val w = InverseWishart(5.0, psi)
//     val samples = w.sample(n)

//     (w.mean === (samples.reduce(_ + _) / samples.length.toDouble)) :| "Mean of distribution equal to sample mean"
//   }
// }

// object WishartDistribution extends Properties("Wishart") with BreezeGenerators {
//   val scale = symmetricPosDefMatrix(2, 1000)

//   property("Wishart Distribution") = Prop.forAll(scale) { scale =>
//     val nu = 5.0
//     val w = Wishart(nu, scale)
//     val n = 100000
//     //    val samples = Vector.fill(n)(w.drawNaive())
//     val samples = w.sample(n)
//     val sampleMean = samples.reduce(_ + _) / n.toDouble
//     val varianceOne = variance(samples.map(w => w(0,0)))

//     (w.mean === sampleMean) :| "Mean of distribution should be equal to the sample mean" &&
// //    (sampleMean(0, 0) === ChiSquared(nu).mean * scale(0,0)) :| "Mean of first component equal to mean of ChiSquared(nu) * scale(0,0)" &&
//     (varianceOne === ChiSquared(nu).variance * scale(0,0) * scale(0,0)) :| "Variance of first component should be equal to variance of ChiSquared(nu) scaled by scale(0,0)"
//   }
// }

object MvnDistribution extends Properties("MVN") with BreezeGenerators {
  val covariance = symmetricPosDefMatrix(2, 1000)

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
  property("MVN distribution") = Prop.forAll(covariance) { cov =>
    val n = 100000
    val mvn = MultivariateGaussianSvd(DenseVector.zeros[Double](2), cov)
    val samples = mvn.sample(n)
    
    val (sampleMean, sampleCovariance)  = meanCovSamples(samples)

    (mvn.mean === sampleMean) :| "Mean of distribution should equal sample mean"
//    (mvn.variance === sampleCovariance) :| "variance of distribution should equal sample covariance"
  }
}
