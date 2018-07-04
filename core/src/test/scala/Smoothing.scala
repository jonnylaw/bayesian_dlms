import core.dlm.model._
import Dlm._
import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest._
import prop._
import org.scalactic.Equality
import cats.instances.vector._
import cats.syntax.traverse._

trait SmoothedData {
  val model = Dlm.polynomial(1)
  val p = DlmParameters(
    v = DenseMatrix(3.0),
    w = DenseMatrix(1.0),
    m0 = DenseVector(0.0),
    c0 = DenseMatrix(1.0)
  )

  val data = Vector(
    Data(1.0, DenseVector(Some(4.5))),
    Data(2.0, DenseVector(Some(3.0))),
    Data(3.0, DenseVector(Some(6.3))),
    Data(4.0, DenseVector[Option[Double]](None)),
    Data(5.0, DenseVector(Some(10.1))),
    Data(7.0, DenseVector(Some(15.2)))
  )

  // tolerance
  implicit val tol = 1e-4

  val filtered = KalmanFilter.filter(model, data, p, KalmanFilter.advanceState(p, model.g))
  val smoothed = Smoothing.backwardsSmoother(model)(filtered)
}

class SmoothingTest extends FunSuite with Matchers with SmoothedData {
  test(
    "Smoothed should be length of the data + 1, including initial state at t = 0") {
    assert(smoothed.size == data.size + 1)
  }

  val s7 = smoothed.last.mean(0)
  val m5 = filtered(5).mt(0)
  val c5 = filtered(5).ct(0, 0)
  val r7 = filtered.last.rt(0, 0)
  val a7 = filtered.last.at(0)
  val s5 = m5 + c5 * 1 / r7 * (s7 - a7)

  val S7 = smoothed.last.covariance(0, 0)
  val S5 = c5 - c5 * c5 * 1 / (r7 * r7) * (r7 - S7)

  test(
    "The initial smoothing mean and covariance should be equal to the last filtered mean and covariance") {
    assert(filtered.last.mt(0) === s7 +- tol)
    assert(filtered.last.ct(0, 0) === S7 +- tol)
  }

  test(
    "The first step of the smoothing algorithm to calculate s5 and S5 from s7 and S7") {
    assert(smoothed(5).mean(0) === s5 +- tol)
    assert(smoothed(5).covariance(0, 0) === S5 +- tol)
  }

  val m4 = filtered(4).mt(0)
  val c4 = filtered(4).ct(0, 0)
  val r5 = filtered(5).rt(0, 0)
  val a5 = filtered(5).at(0)

  val s4 = m4 + c4 * 1 / r5 * (s5 - a5)
  val S4 = c4 - c4 * c4 * 1 / (r5 * r5) * (r5 - S5)

  test("Second smoothing step") {
    assert(smoothed(4).mean(0) === s4 +- tol)
    assert(smoothed(4).covariance(0, 0) === S4 +- tol)
  }
}
