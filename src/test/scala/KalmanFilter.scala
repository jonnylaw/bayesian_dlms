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

class KfSpec extends PropSpec with GeneratorDrivenPropertyChecks with Matchers with BreezeGenerators {
  def linearSystem(dim: Int) = for {
    qt <- symmetricPosDefMatrix(dim, 100)
    rt <- denseVector(2)
  } yield (rt.t, qt)

  property("Solution to linear system") {
    forAll (linearSystem(2)) { case (rt, qt) =>
      implicit val tol = 1e-2
      val naive = rt * inv(qt)
      val better = (qt.t \ rt.t).t

      assert(better.t === naive.t)
    }
  }

  val params = for {
    v <- smallDouble
    w <- symmetricPosDefMatrix(2, 100)
    m0 = DenseVector.zeros[Double](2)
    c0 = DenseMatrix.eye[Double](2) * 100.0
  } yield Parameters(DenseMatrix(v), w, m0, c0)

  val mod = Dlm.polynomial(2)

  def observations(p: Parameters) = 
    Dlm.simulateRegular(0, mod, p).steps.take(100).map(_._1).toArray

  property("Kalman Filter State should be one length observations + 1") {
    forAll (params) { p =>
      val data = observations(p)
      val filtered = KalmanFilter.filter(mod, data, p)

      assert(filtered.size === (data.size + 1))
      assert(filtered.map(_.time).tail === data.map(_.time))
    }
  }

  property("Backward Sampling is the length of the filtered state and contains the same times") {
    forAll (params) { p =>
      val data = observations(p)
      val filtered = KalmanFilter.filter(mod, data, p)
      val sampled = Smoothing.sample(mod, filtered, p.w)

      assert(sampled.size === filtered.size)
      assert(sampled.map(_._1) === filtered.map(_.time))
    }
  }
}

class KalmanFilterTest extends FunSuite with Matchers with BreezeGenerators {
  val model = Dlm.polynomial(1)
  val p = Dlm.Parameters(
    v = DenseMatrix(3.0),
    w = DenseMatrix(1.0),
    m0 = DenseVector(0.0),
    c0 = DenseMatrix(1.0)
  )

  val data = Array(
    Data(1.0, DenseVector(Some(4.5))),
    Data(2.0, DenseVector(Some(3.0))),
    Data(3.0, DenseVector(Some(6.3))),
    Data(4.0, DenseVector[Option[Double]](None)),
    Data(5.0, DenseVector(Some(10.1))),
    Data(7.0, DenseVector(Some(15.2)))
  )

  val y1 = data.head
  val (a1, r1) = KalmanFilter.advanceState(model.g, p.m0, p.c0, 1, p.w)
  val (f1, q1, m1, c1, _) = KalmanFilter.updateState(model.f, a1, r1, y1, p.v, 0.0)
  val e1 = KalmanFilter.flattenObs(y1.observation) - f1
  val k1 = r1 * inv(q1)

  // tolerance
  implicit val tol = 1e-4

  test("advance state for first order model be a1 = m0, R1 = C0 + W") {
    assert(a1 === p.m0)
    assert(r1 === p.c0 + p.w)
  }

  test("one step prediction for first order model should be, f1 = a1, Q1 = R1 + V") {
    assert(f1 === a1)
    assert(q1 === r1 + p.v)
  }

  test("update for first order model should be, m1 = a1 + k1 * e1, c1 = r1 - k1 * r1") {
    assert(m1 === a1 + k1 * e1)
    assert(c1 === r1 - k1 * r1)
  }

  val state1 = KalmanFilter.State(1, m1, c1, a1, r1, Some(f1), Some(q1), 0.0)
  val filterOne = KalmanFilter.step(model, p)(state1, data(1))

  test("time step 2") {
    assert(filterOne.at(0) === 1.8)
    assert(filterOne.rt(0,0) === 2.2)

    assert(filterOne.y.get(0) === 1.8)
    assert(filterOne.cov.get(0,0) === 5.2)

    assert(filterOne.mt(0) === 2.307692 +- tol)
    assert(filterOne.ct(0,0) === 1.269231 +- tol)
  }

  val filterTwo = KalmanFilter.step(model, p)(filterOne, data(2))

  test("time step 3") {
    assert(filterTwo.at(0) === 2.307692 +- tol)
    assert(filterTwo.rt(0,0) === 2.269231 +- tol)

    assert(filterTwo.y.get(0) === 2.307692 +- tol)
    assert(filterTwo.cov.get(0,0) === 5.269231 +- tol)

    assert(filterTwo.mt(0) === 4.027007 +- tol)
    assert(filterTwo.ct(0,0) === 1.291971 +- tol)
  }

  val filterThree = KalmanFilter.step(model, p)(filterTwo, data(3))

  test("time step 4, missing data") {
    assert(filterThree.at(0) === 4.027007 +- tol)
    assert(filterThree.rt(0,0) === 2.291971 +- tol)

    assert(filterThree.y.get(0) === 4.027007 +- tol)
    assert(filterThree.cov.get(0,0) === 5.291971 +- tol)

    assert(filterThree.mt(0) === 4.027007 +- tol)
    assert(filterThree.ct(0,0) === 2.291971 +- tol)
  }

  val filterFour = KalmanFilter.step(model, p)(filterThree, data(4))

  test("time step, t = 5") {
    assert(filterFour.at(0) === 4.027007 +- tol)
    assert(filterFour.rt(0,0) === 3.291971 +- tol)

    assert(filterFour.y.get(0) === 4.027007 +- tol)
    assert(filterFour.cov.get(0,0) === 6.291971 +- tol)

    assert(filterFour.mt(0) === 7.204408 +- tol)
    assert(filterFour.ct(0,0) === 1.569606 +- tol)
  }

  val filterFive = KalmanFilter.step(model, p)(filterFour, data(5))

  test("time step, t = 7, skip an observation without encoding") {
    assert(filterFive.at(0) === 7.204408 +- tol)
    assert(filterFive.rt(0,0) === 3.569606 +- tol)

    assert(filterFive.y.get(0) === 7.204408 +- tol)
    assert(filterFive.cov.get(0,0) === 6.569606 +- tol)

    assert(filterFive.mt(0) === 11.54883 +- tol)
    assert(filterFive.ct(0,0) === 1.630055 +- tol)
  }

  // val filtered = KalmanFilter.filter(model, data, p)

  // test("Full kalman Filter is equivalent to all steps") {
  //   assert(filtered.head === filter
  // }

  test("Missing Matrix") {
    val obs = DenseVector(Some(1), None, Some(1), None, Some(5), None)
    val initMatrix = (t: Double) => DenseMatrix.ones[Double](10, 6)

    val fm = KalmanFilter.missingF(initMatrix, 1.0, obs)

    assert(fm.cols === obs.data.flatten.size)
    assert(fm.rows === 10)
  }
}
