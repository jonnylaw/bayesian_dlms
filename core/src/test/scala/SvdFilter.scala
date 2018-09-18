import dlm.core.model._
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

  property("SVD of symmetric positive definite matrix is U * D * U.t") {
    forAll(symmetricPosDefMatrix(3, 100)) { m =>
      implicit val tol = 1e-4

      val root = svd(m)
      val u = root.rightVectors.t
      val d = SvdFilter.makeDMatrix(u.cols, u.cols, root.singularValues)
      assert(m === u * d * u.t)
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

class SvdFilterTest extends FunSuite with Matchers with BreezeGenerators {
  implicit val tol = 1e-2

  val model = Dlm.polynomial(1) |*| Dlm.polynomial(1)
  val p = DlmParameters(
    v = diag(DenseVector(3.0, 3.0)),
    w = diag(DenseVector(1.0, 1.0)),
    m0 = DenseVector.fill(2)(0.0),
    c0 = diag(DenseVector(1.0, 1.0))
  )

  val data = Vector(
    Data(1.0, DenseVector(Some(4.5), Some(4.5))),
    Data(2.0, DenseVector(Some(3.0), Some(3.0))),
    Data(3.0, DenseVector(Some(6.3), Some(6.3))),
    Data(4.0, DenseVector[Option[Double]](None, None)),
    Data(5.0, DenseVector(Some(10.1), None)), // partially observed
    Data(7.0, DenseVector(Some(15.2), Some(15.2)))
  )

  val filteredTest = KalmanFilter.filterDlm(model, data, p)
  val filterSvdTest = SvdFilter.filterDlm(model, data, p)

  val root = svd(p.c0)
  val uc = root.rightVectors.t
  val dc = root.singularValues
  val svdkf = SvdFilter(SvdFilter.advanceState(p, model.g))
  val (a1svd, dr, ur) = SvdFilter.advState(model.g, 1.0, p.m0, dc, uc, p.w)
  val r1svd = (diag(dr) * ur).t * (diag(dr) * ur)

  test("Svd filter should advance the state correctly") {
    implicit val tol = 1e-3

    assert(a1svd === p.m0)
    assert(r1svd === p.c0 + p.w)
  }

  val f1svd = svdkf.oneStepForecast(model.f, a1svd, 1.0)

  test("Svd Filter should perform a one step prediction") {
    assert(f1svd === filteredTest.head.at)
  }

  test("Svd filter should return the same values as the Kalman Filter") {
    for {
      i <- 0 until data.size
    } assert(filteredTest(i).mt === (filterSvdTest(i).mt))
  }

  val covs = filterSvdTest.map(s => (diag(s.dc) * s.uc.t).t * (diag(s.dc) * s.uc.t))
  test("svd filter covariances should the same values as the Kalman Filter") {
    for {
      i <- 0 until data.size
    } assert(filteredTest(i).ct === (covs(i)))
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
    val sampled = Smoothing.ffbsDlm(model, data, p).draw
    val svdSampled = SvdSampler.ffbs(model, data, p, SvdFilter.advanceState(p, model.g)).draw

    assert(sampled.map(_.sample.size).sum === svdSampled.map(_._2.size).sum)
    sampled.map(_.time) should contain allElementsOf svdSampled.map(_._1)
  }
}
