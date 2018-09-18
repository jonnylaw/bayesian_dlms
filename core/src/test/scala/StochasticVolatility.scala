import dlm.core.model._
import org.scalatest._
import org.scalactic.Equality
import breeze.linalg.{diag, DenseVector, DenseMatrix}
import breeze.stats.distributions._

class StochVolTest extends FunSuite with Matchers with BreezeGenerators {
  implicit val tol = 1e-2

  // simulate data
  val p = SvParameters(0.8, 1.0, 0.2)
  val model = Dlm.autoregressive(p.phi)
  val params = StochasticVolatility.ar1DlmParams(p)
  val sims = StochasticVolatility.simulate(p).
    steps.
    take(100).
    toVector.
    map { case (t, y, a) => Data(t, DenseVector(y)) }

  val filteredTest = KalmanFilter(FilterAr.advanceState(p)).
    filter(model, sims, params)
  val filterSvdTest = SvdFilter(FilterAr.advanceStateSvd(p)).
    filter(model, sims, params)

  test("Svd filter AR(1) should return the same values as the Kalman Filter") {
    for {
      i <- 0 until sims.size
    } assert(filteredTest(i).mt === (filterSvdTest(i).mt))
  }

  val covs = filterSvdTest.map(s => (diag(s.dc) * s.uc.t).t * (diag(s.dc) * s.uc.t))
  test("svd filter AR(1) covariances should the same values as the Kalman Filter") {
    for {
      i <- 0 until sims.size
    } assert(filteredTest(i).ct === (covs(i)))
  }

  // TODO: calculate AR(1) filter "by hand" to test SVD Filter
  val data = Vector(
    Data(1.0, DenseVector(Some(4.5), Some(4.5))),
    Data(2.0, DenseVector(Some(3.0), Some(3.0))),
    Data(3.0, DenseVector(Some(6.3), Some(6.3))),
    Data(4.0, DenseVector[Option[Double]](None, None)),
    Data(5.0, DenseVector(Some(10.1), None)), // partially observed
    Data(7.0, DenseVector(Some(15.2), Some(15.2)))
  )

  val mod = Dlm.polynomial(2)
  val firstOrderParams = DlmParameters(
    DenseMatrix.eye[Double](1),
    DenseMatrix.eye[Double](2) * 2.0,
    DenseVector.zeros[Double](2),
    DenseMatrix.eye[Double](2) * 10.0)
  
  val firstOrderSims = Dlm.simulateRegular(mod, firstOrderParams, 1.0).
    steps.take(2).
    toVector.
    map(_._1)

  val sampledState = Smoothing.ffbsDlm(mod, firstOrderSims, firstOrderParams).draw

  test("extract state should be inverse to combine states") {
    val extracted = for {
      i <- Vector.range(0, 2)
      st = FactorSv.extractState(sampledState, i)
    } yield st

    val combined = FactorSv.combineStates(extracted)

    assert(combined.map(_.sample) === sampledState.map(_.sample))
    assert(combined.map(_.time) === sampledState.map(_.time))
    assert(combined.map(x => diag(x.cov)) === sampledState.map(x => diag(x.cov)))
    assert(combined.map(x => diag(x.rt1)) === sampledState.map(x => diag(x.rt1)))
  }

  implicit def seqEq(implicit tol: Double) =
    new Equality[Seq[Double]] {
      def areEqual(x: Seq[Double], b: Any) = b match {
        case y: Seq[Double] =>
          (x, y).zipped.forall { case (a, b) => math.abs(a - b) < tol }
        case _ => false
      }
    }

  test("log-sum-exp should produce equivalent normalised weights") {
    val weights = Uniform(0, 1).sample(100)
    val logWeights = weights.map(math.log)
    val max = logWeights.max
    val w1 = logWeights.map(w => math.exp(w - max))

    def normalise(w: Seq[Double]) = {
      val total = w.sum
      w.map(_ / total)
    }

    assert(normalise(w1) === normalise(weights))
  }
}
