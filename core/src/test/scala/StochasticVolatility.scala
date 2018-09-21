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
  val params = StochVolKnotsMultivariate.ar1DlmParams(p)
  val sims = StochasticVolatility.simulate(p).
    steps.
    take(10).
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

  val ys = sims.map(d => (d.time, d.observation(0)))
  val filtered = FilterAr.filterUnivariate(ys, Vector.fill(ys.size)(1.0), p)

  test("Univariate Kalman filter is the same as the Kalman Filter") {
    for {
      i <- 0 until sims.size
      means = filtered.map(_.mt)
      mvMeans = filteredTest.map(_.mt(0))
    } assert(means(i) === (mvMeans(i)) +- 1e-3)

    for {
      i <- 0 until sims.size
      covs = filtered.map(_.ct)
      mvCovs = filteredTest.map(_.ct(0,0))
    } assert(covs(i) === (mvCovs(i)) +- 1e-3)
  }

  val sampled = Smoothing.sample(model, filteredTest, FilterAr.backStep(p))
  val sampledUni = FilterAr.univariateSample(p, filtered).draw

  test("Univariate Sampler is the same size MV Sampler") {
    assert(sampled.size === sampledUni.size)
  }

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

  test("Knots should remain unchanged in a single sample") {
    import StochasticVolatilityKnots._

    val arsims = Dlm.simulateRegular(model, params, 1.0).steps.take(1000)
    val ys = arsims.map(_._1).map(d => (d.time, d.observation(0))).toVector
    val alphas = FilterAr.ffbs(p, ys, Vector.fill(ys.size)(1.0)).draw

    val knots = sampleKnots(10, 100)(ys.size).draw

    val sampled = sampleState(
        ffbsAr, filterAr, sampleAr)(ys, p, knots, alphas.toArray).toVector

    // extract the knots
    val resampled = for {
      i <- knots.tail.init
      sample = sampled(i)
    } yield (sample.time, sample.sample)

    val initialSample = for {
      i <- knots.tail.init
      sample = alphas(i)
    } yield (sample.time, sample.sample)

    assert(resampled === initialSample)
  }
}
