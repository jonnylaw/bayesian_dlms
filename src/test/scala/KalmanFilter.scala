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
    Dlm.simulate(0, mod, p).steps.take(100).map(_._1).toArray

  property("Kalman Filter State should be one length observations + 1") {
    forAll (params) { p =>
      val data = observations(p)
      val filtered = KalmanFilter.kalmanFilter(mod, data, p)

      assert(filtered.size === (data.size + 1))
      assert(filtered.map(_.time).tail === data.map(_.time))
    }
  }

  property("Backward Sampling is the length of the filtered state and contains the same times") {
    forAll (params) { p =>
      val data = observations(p)
      val filtered = KalmanFilter.kalmanFilter(mod, data, p)
      val sampled = Smoothing.backwardSampling(mod, filtered, p.w)

      assert(sampled.size === filtered.size)
      assert(sampled.map(_._1) === filtered.map(_.time))
    }
  }

  // TODO: Add Exact unit tests for a hand-calculated Kalman Filter
}
