import core.dlm.model._
import org.scalatest._
import prop._
import org.scalactic.Equality
import breeze.linalg.{diag, DenseVector, DenseMatrix, svd, eigSym, inv}
import breeze.stats.distributions.MultivariateGaussian
import cats.instances.vector._
import cats.syntax.traverse._

class SvdKfSpec
    extends PropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers
    with BreezeGenerators {
  property("Square of square root matrix is the matrix") {
    forAll(symmetricPosDefMatrix(2, 100)) { m =>
      implicit val tol = 1e-4

      val sqrt = SvdFilter.sqrtSvd(m)

      assert(m === sqrt.t * sqrt)
    }
  }

  property("sqrtInvSym works") {
    forAll(symmetricPosDefMatrix(2, 100)) { m =>
      implicit val tol = 1e-4

      val sqrt = SvdFilter.sqrtInvSvd(m)

      assert(m === inv(sqrt.t * sqrt))
    }
  }

  property("SVD of a symmetric matrix is u * diag(d) * u.t") {
    forAll(symmetricPosDefMatrix(3, 100)) { m =>
      implicit val tol = 1e-4

      val root = svd(m)
      val u = root.rightVectors.t
      val d = root.singularValues

      assert(m === u * diag(d) * u.t)
    }
  }

  property("Eigenvalue decomposition is U * diag(d) * U.t") {
    forAll(symmetricPosDefMatrix(3, 100)) { m =>
      implicit val tol = 1e-4

      val root = eigSym(m)
      val u = root.eigenvectors
      val d = root.eigenvalues

      assert(m === u * diag(d) * u.t)
    }
  }

  property("SVD of a matrix U * D * Vt") {
    forAll(denseMatrix(3, 2)) { m =>
      implicit val tol = 1e-4

      val root = svd(m)
      val u = root.leftVectors
      val vt = root.rightVectors
      val d = SvdFilter.makeDMatrix(u.cols, vt.cols, root.singularValues)
      assert(m === u * d * vt)
    }
  }

  ignore("rnorm should sample from the multivariate normal distribution") {
    forAll(symmetricPosDefMatrix(2, 10)) { m =>
      implicit val tol = 1e-1

      val n = 1000000
      val root = eigSym(m)
      val mean = DenseVector.zeros[Double](2)
      val rnormed = Dglm.meanCovSamples(
        SvdSampler
          .rnorm(mean, root.eigenvalues.map(math.sqrt), root.eigenvectors)
          .sample(n))
      val breezed = Dglm.meanCovSamples(MultivariateGaussian(mean, m).sample(n))

      assert(rnormed._1 === breezed._1)
      assert(rnormed._2 === breezed._2)
    }
  }
}

class SvdFilterTest extends FlatSpec with Matchers with BreezeGenerators {
  val model = Dlm.polynomial(2)
  val p = DlmParameters(
    DenseMatrix(3.0),
    diag(DenseVector(2.0, 1.0)),
    DenseVector(0.0, 0.0),
    diag(DenseVector(100.0, 100.0))
  )

  val data =
    Dlm.simulateRegular(model, p, 1.0).steps.take(3).toVector.map(_._1)

  // tolerance
  implicit val tol = 1e-2

  val filtered = KalmanFilter.filter(model, data, p)
  val svdFiltered = SvdFilter.filter(model, data, p)
  val covs =
    svdFiltered.map(s => (diag(s.dc) * s.uc.t).t * (diag(s.dc) * s.uc.t))

  "Svd Filter" should "produce the same filtered state as the Kalman Filter" in {
    filtered.map(_.mt) should contain allElementsOf (svdFiltered.map(_.mt))
    filtered.map(_.ct) should contain allElementsOf covs
  }

  val filterArray = FilterArray.filterSvd(model, data, p)
  val covsArray =
    filterArray.map(s => (diag(s.dc) * s.uc.t).t * (diag(s.dc) * s.uc.t))

  "Filter Array Svd" should "produce the same state as the Kalman Filter" in {
    filtered.map(_.mt) should contain allElementsOf (filterArray.map(_.mt))
    filtered.map(_.ct) should contain allElementsOf covsArray
  }

  val model2 = Dlm.polynomial(1) |*| Dlm.polynomial(1)
  val p2 = DlmParameters(
    v = diag(DenseVector(3.0, 3.0)),
    w = diag(DenseVector(1.0, 1.0)),
    m0 = DenseVector.fill(2)(0.0),
    c0 = diag(DenseVector(1.0, 1.0))
  )

  val data2 = Vector(
    Dlm.Data(1.0, DenseVector(Some(4.5), Some(4.5))),
    Dlm.Data(2.0, DenseVector(Some(3.0), Some(3.0))),
    Dlm.Data(3.0, DenseVector(Some(6.3), Some(6.3))),
    Dlm.Data(4.0, DenseVector[Option[Double]](None, None)),
    Dlm.Data(5.0, DenseVector(Some(10.1), None)), // partially observed
    Dlm.Data(7.0, DenseVector(Some(15.2), Some(15.2)))
  )

  val filteredTest = KalmanFilter.filter(model2, data2, p2)
  val filterSvdTest = SvdFilter.filter(model2, data2, p2)
  "Svd Filter" should "handle missing observations" in {
    filteredTest.map(_.mt) should contain allElementsOf (filterSvdTest.map(
      _.mt))
  }
}

class SvdSamplerTest extends FlatSpec with Matchers with BreezeGenerators {
  val model = Dlm.polynomial(2)
  val p = DlmParameters(
    v = diag(DenseVector(3.0)),
    w = diag(DenseVector(1.0, 1.0)),
    m0 = DenseVector.fill(2)(0.0),
    c0 = DenseMatrix.eye[Double](2) * 0.5
  )

  val data =
    Dlm.simulateRegular(model, p, 1.0).steps.take(10).toVector.map(_._1)

  // tolerance
  implicit val tol = 1e-2

  "Svd Sampler" should "produce the same size state as smoothing sampler" in {
    val sampled = Smoothing.ffbs(model, data, p).draw
    val svdSampled = SvdSampler.ffbs(model, data, p).draw

    assert(sampled.map(_._2.size).sum === svdSampled.map(_._2.size).sum)
    sampled.map(_._1) should contain allElementsOf svdSampled.map(_._1)
  }
}
