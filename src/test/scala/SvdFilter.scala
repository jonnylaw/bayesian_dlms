import dlm.model._
import org.scalatest._
import prop._
import org.scalactic.Equality

class SvdKfSpec extends PropSpec with GeneratorDrivenPropertyChecks with Matchers with BreezeGenerators {
   property("Square of square root matrix") {
    forAll (symmetricPosDefMatrix(2, 100)) { m =>
      implicit val tol = 1e-4

      val sqrt = SvdFilter.sqrtSym(m)

      assert(m === sqrt.t * sqrt)
    }
  }
}

// class SvdMvnDistribution extends PropSpec with GeneratorDrivenPropertyChecks with Matchers with BreezeGenerators {  
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
