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
  val sims = StochasticVolatility.simulate(p).
    steps.
    take(1000).
    toVector.
    map { case (t, y, a) => Data(t, DenseVector(y)) }

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

  import StochasticVolatilityKnots._

  val params = DlmParameters(1.0, 0.8, 0.0, 1.0)
  val arsims = Dlm.simulateRegular(model, params, 1.0).
    steps.
    take(1000).
    map(_._1).
    map(d => (d.time, d.observation(0))).
    toVector
  val alphas = FilterAr.ffbs(p, arsims, Vector.fill(arsims.size)(1.0)).draw

  val knots = sampleKnots(10, 100, arsims.size).draw

  val sampledAr = sampleState(
    ffbsAr, filterAr, sampleAr)(arsims, p, knots, alphas.toArray).toVector

  test("Knots should remain unchanged in a single sample") {
    // extract the knots
    val resampled = for {
      i <- knots.tail.init
      sample = sampledAr(i)
    } yield (sample.time, sample.sample)

    val initialSample = for {
      i <- knots.tail.init
      sample = alphas(i)
    } yield (sample.time, sample.sample)

    assert(resampled === initialSample)
  }

  test("Things in between the knots should be changed") {
    assert(sampledAr !== alphas)
  }

  test("The final sampled state should be altered") {
    assert(sampledAr.last !== alphas.last)
  }

  test("The initial sampled state should be altered") {
    assert(sampledAr.head !== alphas.head)
  }

  val sampledArFold = sampleStateFold(
    ffbsAr, filterAr, sampleAr)(arsims, p, knots, alphas.toArray).toVector

  // test("Folded state is the same length as initial state") {
  //   assert(sampledArFold.size === alphas.size)
  // }

  // test("Folded knots should remain unchanged in a single sample") {
  //   // extract the knots
  //   val resampled = for {
  //     i <- knots.tail.init
  //     sample = sampledArFold(i)
  //   } yield (sample.time, sample.sample)

  //   val initialSample = for {
  //     i <- knots.tail.init
  //     sample = alphas(i)
  //   } yield (sample.time, sample.sample)

  //   assert(resampled === initialSample)
  // }

  // test("The final sampled state folded should be altered") {
  //   assert(sampledArFold.last !== alphas.last)
  // }

  // test("The initial sampled state folded should be altered") {
  //   assert(sampledArFold.head !== alphas.head)
  // }
}
