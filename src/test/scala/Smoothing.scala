 import dlm.model._
import Dlm._
import breeze.linalg.{DenseMatrix, DenseVector, diag, inv}
import breeze.stats.distributions.{ChiSquared, Gamma}
import breeze.stats.covmat
import breeze.stats.{meanAndVariance, variance, mean}
import org.scalatest._
import prop._
import org.scalacheck.Gen
import org.scalacheck.Arbitrary
import Arbitrary.arbitrary
import org.scalactic.Equality
import org.scalactic.Equality

class SmoothingTest extends FunSuite with Matchers with BreezeGenerators {
  val model = Dlm.polynomial(1)
  val p = Dlm.Parameters(
    v = DenseMatrix(3.0),
    w = DenseMatrix(1.0),
    m0 = DenseVector(0.0),
    c0 = DenseMatrix(1.0)
  )

  val data = Array(
    Data(1.0, Some(DenseVector(4.5))),
    Data(2.0, Some(DenseVector(3.0))),
    Data(3.0, Some(DenseVector(6.3))),
    Data(4.0, None),
    Data(5.0, Some(DenseVector(10.1))),
    Data(7.0, Some(DenseVector(15.2)))
  )

  // tolerance
  implicit val tol = 1e-4

  val filtered = KalmanFilter.filter(model, data, p)

  val smoothed = Smoothing.backwardsSmoother(model)(filtered)

  val s6 = smoothed.last.mean(0)
  val m5 = filtered(4).mt(0)
  val c5 = filtered(4).ct(0, 0)
  val r6 = filtered.last.rt(0,0)
  val a6 = filtered.last.at(0)
  val s5 = m5 + c5 * c5 * 1 / (r6 * r6) * (s6 - a6)

  val S6 = smoothed.last.covariance(0,0)
  val S5 = c5 - c5 * c5 * 1 / (r6 * r6) * (r6 - S6)

  test("The initial smoothing mean and covariance should be equal to the last filtered mean and covariance") {
    assert(filtered.last.mt(0) === s6 +- tol)
    assert(filtered.last.ct(0,0) === S6 +- tol)
  }

  test("A single step of the smoothing algorithm") {
    assert(smoothed(5).mean(0) === s5 +- tol)
    assert(smoothed(5).covariance(0,0) === S5 +- tol)
  }
}
