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

  val sortedState = filtered.sortWith(_.time > _.time)
  val last = sortedState.head
  val lastTime = last.time
  val init = Smoothing.SmoothingState(lastTime, last.mt, last.ct, last.at, last.rt)

  val smoothed = Smoothing.backwardsSmoother(model)(filtered)
  val smoothOne = Smoothing.smoothStep(model)(init, sortedState(1))

  test("A single step of the smoothing algorithm") {
    assert(smoothOne.mean(0) === 6.239211 +- tol)
    assert(smoothOne.covariance(0,0) === 0.8348942 +- tol)
  }
}
