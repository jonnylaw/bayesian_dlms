import dlm.model._
import org.scalatest._
import prop._
import org.scalactic.Equality
import breeze.linalg.{diag, DenseVector, svd, eigSym, inv}
import breeze.stats.distributions.MultivariateGaussian

class SvdKfSpec extends PropSpec with GeneratorDrivenPropertyChecks with Matchers with BreezeGenerators {
   property("Square of square root matrix is the matrix") {
    forAll (symmetricPosDefMatrix(2, 100)) { m =>
      implicit val tol = 1e-4

      val sqrt = SvdFilter.sqrtSvd(m)

      assert(m === sqrt.t * sqrt)
    }
  }

  property("sqrtInvSym works") {
    forAll (symmetricPosDefMatrix(2, 100)) { m =>
      implicit val tol = 1e-4

      val sqrt = SvdFilter.sqrtInvSym(m)

      assert(m === inv(sqrt.t * sqrt))
    }
  }

  property("SVD of symmetric matrix is u * diag(d) * u.t") {
    forAll (symmetricPosDefMatrix(3, 100)) { m =>
      implicit val tol = 1e-4

      val root = svd(m)
      val u = root.rightVectors.t
      val d = root.singularValues

      assert(m === u * diag(d) * u.t)
    }
  }

  property("Eigenvalue decomposition is U * diag(d) * U.t") {
    forAll (symmetricPosDefMatrix(3, 100)) { m =>
      implicit val tol = 1e-4

      val root = eigSym(m)
      val u = root.eigenvectors
      val d = root.eigenvalues

      assert(m === u * diag(d) * u.t)
    }
  }

  ignore("rnorm should sample from the multivariate normal distribution") {
    forAll (symmetricPosDefMatrix(2, 10)) { m =>
      implicit val tol = 1e-1

      val n = 1000000
      val root = eigSym(m)
      val mean = DenseVector.zeros[Double](2)
      val rnormed = Dglm.meanCovSamples(SvdSampler.rnorm(mean,
        root.eigenvalues.map(math.sqrt), root.eigenvectors).sample(n))
      val breezed = Dglm.meanCovSamples(MultivariateGaussian(mean, m).sample(n))
      
      assert(rnormed._1 === breezed._1)
      assert(rnormed._2 === breezed._2)
    }
  }
}

class SvdFilterSamplerTest extends FlatSpec with Matchers with BreezeGenerators {
  val model = Dlm.polynomial(1)
  val p = Dlm.Parameters(
    v = diag(DenseVector(3.0)),
    w = diag(DenseVector(1.0)),
    m0 = DenseVector.fill(1)(0.0),
    c0 = diag(DenseVector(1.0))
  )

  val data = Dlm.simulateRegular(0, model, p, 1.0).
    steps.
    take(100).
    toVector.
    map(_._1)

  // tolerance
  implicit val tol = 1e-2

  val filtered = KalmanFilter.filter(model, data, p)
  val svdFiltered = SvdFilter.filter(model, data, p)
  val covs = svdFiltered.map(s => s.uc * s.dc.t * s.dc * s.uc.t)

  "Svd Filter" should "produce the same filtered state as the Kalman Filter" in {
    filtered.map(_.mt) should contain allElementsOf (svdFiltered.map(_.mt))
    filtered.map(_.ct) should contain allElementsOf covs
  }

  "Svd Sampler" should "produce the same size state as smoothing sampler" in {
    val sampled = Smoothing.sample(model, filtered, p.w)
    val svdSampled = SvdSampler.sample(model, p.w, svdFiltered)

    assert(sampled.map(_._2.size) == svdSampled.map(_._2.size))
    sampled.map(_._1) should contain allElementsOf svdSampled.map(_._1)
  }

  val filterArray = GibbsSampling.filterArray(model, data, p)
  val covsArray = filterArray.map(s => s.uc * s.dc.t * s.dc * s.uc.t)

  "Filter Array" should "produce the same filtered state as the Kalman Filter" in {
    filtered.map(_.mt) should contain allElementsOf (filterArray.map(_.mt))
    filtered.map(_.ct) should contain allElementsOf covsArray
  }
}
