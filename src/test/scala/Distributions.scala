import dlm.model._
import breeze.linalg._
import breeze.stats.distributions._
import breeze.stats.meanAndVariance
import org.scalatest._
import prop._
import org.scalactic.Equality
import org.scalacheck.Gen
import Dglm._

trait BreezeGenerators {
  val denseVector = (n: Int) =>
    Gen.containerOfN[Array, Double](n, Gen.choose(-10.0, 10.0)).
    map(a => DenseVector(a))

  val denseMatrix = (n: Int) =>
    Gen.containerOfN[Array, Double](n * n, Gen.choose(-10.0, 10.0)).
    map(a => new DenseMatrix(n, n, a))

  /**
    * Simulate a positive definite matrix with a given condition number
    */
  def symmetricPosDefMatrix(n: Int, c: Double) = {
    if (n > 2) {
      for {
        entries <- Gen.containerOfN[Array, Double](n - 2, Gen.choose(1.0, c))
        d = diag(DenseVector(Array(1.0, c) ++ entries))
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

  implicit def matrixeq(implicit tol: Double) = new Equality[DenseMatrix[Double]] {
    def areEqual(x: DenseMatrix[Double], b: Any) = b match {
      case y: DenseMatrix[Double] =>
        x.data.zip(y.data).
          forall { case (a, b) => math.abs(a - b) < tol  } &&
        y.cols == x.cols &&
        y.rows == x.rows
      case _ => false
    }
  }

  implicit def vectoreq(implicit tol: Double) = new Equality[DenseVector[Double]] {
    def areEqual(x: DenseVector[Double], b: Any) = b match {
      case y: DenseVector[Double] =>
        x.data.zip(y.data).
          forall { case (a, b) => math.abs(a - b) < tol }
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
//         val mv = meanAndVariance(samples)
//         assert(g.mean === mv.mean +- (0.1 * g.mean) )
//         assert(g.variance === mv.variance +- (0.1 * g.variance))
//       }
//     }
//   }
// }

// class InverseWishartDistribution extends PropSpec with GeneratorDrivenPropertyChecks with Matchers with BreezeGenerators { 
//   property("Inverse Wishart Distribution") {
//     forAll(symmetricPosDefMatrix(2, 100)) { (psi: DenseMatrix[Double]) =>
//       implicit val tol = 1.0
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
//       implicit val tol = 1.0
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
//   property("MVN distribution") {
//     forAll(symmetricPosDefMatrix(2, 1000)) { cov =>
//       implicit val tol = 1.0
//       val n = 1000000
//       val mvn = MultivariateGaussianSvd(DenseVector.zeros[Double](2), cov)
//       val samples = mvn.sample(n)
      
//       val (sampleMean, sampleCovariance)  = meanCovSamples(samples)

//       assert(mvn.mean === sampleMean)
//       assert(mvn.variance === sampleCovariance)
//     }
//   }

//   property("MVN SVD should calculate the same log-likelihood as Cholesky") {
//     forAll(symmetricPosDefMatrix(2, 1000)) { cov =>
//       implicit val tol = 1e-2
//       val zero = DenseVector.zeros[Double](2)
//       val y = DenseVector(1.0, 2.0)
//       val svd = MultivariateGaussian(zero, cov).pdf(y)
//       val chol = MultivariateGaussianSvd(zero, cov).pdf(y)

//       assert(svd === chol)
//     }
//   }
// }
