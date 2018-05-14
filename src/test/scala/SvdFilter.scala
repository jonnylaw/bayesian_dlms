import dlm.model._
import org.scalatest._
import prop._
import org.scalactic.Equality
import breeze.linalg.{diag, DenseVector, svd, eigSym, inv}

class SvdKfSpec extends PropSpec with GeneratorDrivenPropertyChecks with Matchers with BreezeGenerators {
   property("Square of square root matrix is the matrix") {
    forAll (symmetricPosDefMatrix(2, 100)) { m =>
      implicit val tol = 1e-4

      val sqrt = SvdFilter.sqrtSym(m)

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
      println("Matrix: ")
      println(m)

      val root = svd(m)
      val u = root.rightVectors.t
      val d = root.singularValues

      assert(m === u * diag(d) * u.t)
    }
  }

  property("Eigenvalue decomposition is U * diag(d) * U.t") {
    forAll (symmetricPosDefMatrix(2, 100)) { m =>
      implicit val tol = 1e-4

      val root = eigSym(m)
      val u = root.eigenvectors
      val d = root.eigenvalues

      assert(m === u * diag(d) * u.t)
    }
  }
}

class SvdFilterSamplerTest extends FlatSpec with Matchers with BreezeGenerators {
  val model = Dlm.seasonal(24, 1)
  val p = Dlm.Parameters(
    v = diag(DenseVector(3.0)),
    w = diag(DenseVector(1.0, 1.0)),
    m0 = DenseVector.fill(2)(0.0),
    c0 = diag(DenseVector(1.0, 1.0))
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

  "Svd Filter" should "produce the same filtered state as the Kalman Filter" in {
    filtered.map(_.mt) should contain allElementsOf (svdFiltered.map(_.mt))
  }

  "Svd Sampler" should "produce the same size state as smoothing sampler" in {
    val sampled = Smoothing.sample(model, filtered, p.w)
    val svdSampled = SvdSampler.sample(model, p.w, svdFiltered)

    assert(sampled.map(_._2.size) == svdSampled.map(_._2.size))
  }
}
