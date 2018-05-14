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

class SvdFilterTest extends FlatSpec with Matchers with BreezeGenerators {
  val model = Dlm.polynomial(1) |*| Dlm.polynomial(1)
  val p = Dlm.Parameters(
    v = diag(DenseVector(3.0, 3.0)),
    w = diag(DenseVector(1.0, 1.0)),
    m0 = DenseVector.fill(2)(0.0),
    c0 = diag(DenseVector(1.0, 1.0))
  )

  val data = Vector(
    Dlm.Data(1.0, DenseVector(Some(4.5), Some(4.5))),
    Dlm.Data(2.0, DenseVector(Some(3.0), Some(3.0))),
    Dlm.Data(3.0, DenseVector(Some(6.3), Some(6.3))), 
    Dlm.Data(4.0, DenseVector[Option[Double]](None, None)),
    Dlm.Data(5.0, DenseVector(Some(10.1), None)),// partially observed
    Dlm.Data(7.0, DenseVector(Some(15.2), Some(15.2)))
  )

  // tolerance
  implicit val tol = 1e-2

  val filtered = KalmanFilter.filter(model, data, p)
  val svdFiltered = SvdFilter.filter(model, data, p)

  filtered.map(_.mt) should contain allElementsOf (svdFiltered.map(_.mt))
}
