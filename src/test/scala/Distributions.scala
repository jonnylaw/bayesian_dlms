import dlm.model._
import breeze.linalg.{DenseMatrix, DenseVector, cond, diag}
import breeze.stats.distributions.{ChiSquared, Gamma}
import breeze.stats.covmat
import breeze.stats.{meanAndVariance, variance, mean}
import org.scalatest._
import prop._
import org.scalacheck.Gen
import org.scalactic.Equality

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

  val smallDouble = Gen.choose(2.0, 10.0)

  implicit def matrixeq = new Equality[DenseMatrix[Double]] {
    def areEqual(x: DenseMatrix[Double], b: Any) = b match {
      case y: DenseMatrix[Double] =>
        x.data.zip(y.data).
          forall { case (a, b) => math.abs(a - b) < 1.0  } &&
        y.cols == x.cols &&
        y.rows == x.rows
      case _ => false
    }
  }

  implicit def vectoreq = new Equality[DenseVector[Double]] {
    def areEqual(x: DenseVector[Double], b: Any) = b match {
      case y: DenseVector[Double] =>
        x.data.zip(y.data).
          forall { case (a, b) => math.abs(a - b) < 0.001 }
      case _ => false
    }
  }
}

// class InverseGammaDistribution extends PropSpec with GeneratorDrivenPropertyChecks with Matchers with BreezeGenerators {
//   property("Inverse Gamma Distribution") {
//     forAll(smallDouble, smallDouble) { (shape: Double, scale: Double) =>
//       whenever (shape > 2.0 && scale > 1.0) {
//         val n = 10000000
//         val g = InverseGamma(shape, scale)
//         val samples = g.sample(n)

//         assert(g.mean === mean(samples) +- (0.1 * g.mean) ) // relative
//         assert(g.variance === variance(samples) +- (0.1 * g.variance))
//       }
//     }
//   }
// }

// class InverseWishartDistribution extends PropSpec with GeneratorDrivenPropertyChecks with Matchers with BreezeGenerators { 
//   property("Inverse Wishart Distribution") {
//     forAll(symmetricPosDefMatrix(2, 100)) { (psi: DenseMatrix[Double]) =>
//       val n = 100000
//       val w = InverseWishart(5.0, psi)
//       val samples = w.sample(n)

//       assert(w.mean === (samples.reduce(_ + _) / samples.length.toDouble))
//     }
//   }
// }

// class WishartDistribution extends PropSpec with GeneratorDrivenPropertyChecks with Matchers with BreezeGenerators {
//   property("Wishart Distribution") {
//     forAll(symmetricPosDefMatrix(2, 100)) { scale =>
//       val n = 1000000
//       val nu = 5.0
//       val w = Wishart(nu, scale)

//       //    val samples = Vector.fill(n)(w.drawNaive())
//       val samples = w.sample(n)
//       val sampleMean = samples.reduce(_ + _) / n.toDouble
//       val varianceOne = variance(samples.map(w => w(0,0)))
//       val chiSqMean = ChiSquared(nu).mean * scale(0,0)
//       val chiSqVar = ChiSquared(nu).variance * scale(0,0) * scale(0,0)

//       assert(w.mean === sampleMean)
//       assert(sampleMean(0, 0) === chiSqMean +- (0.1 * chiSqMean))
//       assert(varianceOne === chiSqVar +- (0.1 * chiSqVar))
//     }
//   }
// }

// class MvnDistribution extends PropSpec with GeneratorDrivenPropertyChecks with Matchers with BreezeGenerators {
//  /**
//     * Calculate the mean and covariance of a sequence of DenseVectors
//     */
//   def meanCovSamples(samples: Seq[DenseVector[Double]]) = {
//     val n = samples.size
//     val m = new DenseMatrix(n, samples.head.size, samples.map(_.data).toArray.transpose.flatten)
//     val sampleMean = samples.reduce(_ + _).map(_ * 1.0/n)
//     val sampleCovariance = covmat.matrixCovariance(m)

//     (sampleMean, sampleCovariance)
//   }
  
//   property("MVN distribution") {
//     forAll(symmetricPosDefMatrix(2, 1000)) { cov =>
//       val n = 100000
//       val mvn = MultivariateGaussianSvd(DenseVector.zeros[Double](2), cov)
//       val samples = mvn.sample(n)
      
//       val (sampleMean, sampleCovariance)  = meanCovSamples(samples)

//       assert(mvn.mean === sampleMean)
//       assert(mvn.variance === sampleCovariance)
//     }
//   }
// }
