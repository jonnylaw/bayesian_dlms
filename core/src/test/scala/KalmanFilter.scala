import dlm.core.model._
import breeze.linalg.{DenseMatrix, DenseVector, diag, inv, cholesky}
import org.scalatest._
import prop._
import org.scalactic.Equality
import cats.instances.vector._
import cats.syntax.traverse._

class KfSpec
    extends PropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers
    with BreezeGenerators {

  def linearSystem(dim: Int) =
    for {
      qt <- symmetricPosDefMatrix(dim, 100)
      rt <- denseVector(2)
    } yield (rt.t, qt)

  property("Solution to linear system") {
    forAll(linearSystem(2)) {
      case (rt, qt) =>
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
  } yield DlmParameters(DenseMatrix(v), w, m0, c0)

  val mod = Dlm.polynomial(2)

  def observations(p: DlmParameters) =
    Dlm.simulateRegular(mod, p, 1.0).steps.take(100).map(_._1).toVector

  property("Kalman Filter State should be the same length as observations") {
    forAll(params) { p =>
      val data = observations(p)
      val filtered = KalmanFilter.filterDlm(mod, data, p)

      assert(filtered.size === data.size)
      assert(filtered.map(_.time) === data.map(_.time))
    }
  }

  property(
    "Backward Sampling is the length of the filtered state and contains the same times") {
    forAll(params) { p =>
      val data = observations(p)
      val filtered = KalmanFilter.filterDlm(mod, data, p)
      val sampled = Smoothing.sampleDlm(mod, filtered, p.w)

      assert(sampled.size === filtered.size)
      assert(sampled.map(_.time) === filtered.map(_.time))
    }
  }

  ignore("Log-likelihood calculation is correct") {
    forAll(params) { p =>
      val data = observations(p)
      val filtered = KalmanFilter.filterDlm(mod, data, p)
      val state = filtered.map(x => (x.time, x.mt))
      val llc = KalmanFilter.logLikelihoodCholesky(mod, state, p.w)
      val ll = KalmanFilter.logLikelihood(mod, state, p.w)
      assert(ll === llc)
    }
  }
}

class KalmanFilterTest extends FunSuite with Matchers with BreezeGenerators {
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

  val y1 = data.head
  val kf = KalmanFilter(KalmanFilter.advanceState(p, model.g))
  val (a1, r1) = KalmanFilter.advState(model.g, p.m0, p.c0, 1, p.w)
  val (f1, q1, m1, c1) =
    kf.updateState(model.f, a1, r1, y1, p.v)
  val e1 = KalmanFilter.flattenObs(y1.observation) - f1
  val k1 = r1 * inv(q1)

  // tolerance
  implicit val tol = 1e-4

  test("advance state for second order model should be a1 = m0, R1 = C0 + W") {
    assert(a1 === p.m0)
    assert(r1 === p.c0 + p.w)
  }

  test(
    "one step prediction for second order model should be, f1 = a1, Q1 = R1 + V") {
    assert(f1 === a1)
    assert(q1 === r1 + p.v)
  }

  test(
    "update for second order model should be, m1 = a1 + k1 * e1, c1 = r1 - k1 * r1") {
    assert(m1 === a1 + k1 * e1)
    assert(c1 === r1 - k1 * r1)
  }

  val state1 = KfState(1, m1, c1, a1, r1, Some(f1), Some(q1))
  val filterOne = kf.step(model, p)(state1, data(1))

  test("time step 2") {
    assert(filterOne.at === DenseVector(1.8, 1.8))
    assert(filterOne.rt === diag(DenseVector(2.2, 2.2)))

    assert(filterOne.ft.get === DenseVector(1.8, 1.8))
    assert(filterOne.qt.get === diag(DenseVector(5.2, 5.2)))

    assert(filterOne.mt === DenseVector(2.307692, 2.307692))
    assert(filterOne.ct === diag(DenseVector(1.269231, 1.269231)))
  }

  val filterTwo = kf.step(model, p)(filterOne, data(2))

  test("time step 3") {
    assert(filterTwo.at === DenseVector(2.307692, 2.307692))
    assert(filterTwo.rt === diag(DenseVector(2.269231, 2.269231)))

    assert(filterTwo.ft.get === DenseVector(2.307692, 2.307692))
    assert(filterTwo.qt.get === diag(DenseVector(5.269231, 5.269231)))

    assert(filterTwo.mt === DenseVector(4.027007, 4.027007))
    assert(filterTwo.ct === diag(DenseVector(1.291971, 1.291971)))
  }

  val filterThree = kf.step(model, p)(filterTwo, data(3))

  test("time step 4, missing data") {
    assert(filterThree.at === DenseVector(4.027007, 4.027007))
    assert(filterThree.rt === diag(DenseVector(2.291971, 2.291971)))

    assert(filterThree.ft.get === DenseVector(4.027007, 4.027007))
    assert(filterThree.qt.get === diag(DenseVector(5.291971, 5.291971)))

    assert(filterThree.mt === DenseVector(4.027007, 4.027007))
    assert(filterThree.ct === diag(DenseVector(2.291971, 2.291971)))
  }

  val filterFour = kf.step(model, p)(filterThree, data(4))

  test("time step, t = 5, partially observed data") {
    assert(filterFour.at === DenseVector(4.027007, 4.027007))
    assert(filterFour.rt === diag(DenseVector(3.291971, 3.291971)))

    assert(filterFour.ft.get === DenseVector(4.027007, 4.027007))
    assert(filterFour.qt.get === diag(DenseVector(6.291971, 6.291971)))

    assert(filterFour.mt === DenseVector(7.204408, 4.027007))
    assert(filterFour.ct === diag(DenseVector(1.569606, 3.291971)))
  }

  val filterFive = kf.step(model, p)(filterFour, data(5))

  test("time step, t = 7, skip an observation without encoding") {
    assert(filterFive.at === DenseVector(7.204408, 4.027007))
    assert(filterFive.rt === diag(DenseVector(3.569606, 5.291971)))

    assert(filterFive.ft.get === DenseVector(7.204408, 4.027007))
    assert(filterFive.qt.get === diag(DenseVector(6.569606, 8.291971)))

    // calculate the update
    // assert(filterFive.mt === 11.54883)
    // assert(filterFive.ct === 1.630055)
  }

  test("Missing Matrix") {
    val obs = DenseVector(Some(1), None, Some(1), None, Some(5), None)
    val initMatrix = (t: Double) => DenseMatrix.ones[Double](10, 6)

    val fm = KalmanFilter.missingF(initMatrix, 1.0, obs)

    assert(fm.cols === obs.data.flatten.size)
    assert(fm.rows === 10)
  }

  // val model = Dlm.polynomial(1)
  // val params =
  // val ys = Dlm.simulateRegular(model, params, 1.0).
  //   steps.take(100).map(_._1).toVector

  // test("Univariate Kalman Filter should be equivalent to multivariate Kalman Filter") {
  //   val uniFiltered = KalmanFilter.univariateKf(ys.map(d => (d.time, d.observation)), )
  // }
}
