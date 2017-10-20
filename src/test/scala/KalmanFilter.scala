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
  def matrices(dim: Int) = for {
    qt <- symmetricPosDefMatrix(1, 100)
    rt <- symmetricPosDefMatrix(dim, 100)
  } yield (rt, qt)

  property("Solution to linear system") {
    forAll (matrices(2)) { case (rt, qt) =>
      val mod = Dlm.polynomial(2)
      val time = 1
      val kalmanGainNaive = rt * (mod.f(time) * inv(qt))
      val kalmanGainBetter = (qt.t \ (mod.f(time).t * rt.t)).t

      assert(kalmanGainBetter === kalmanGainNaive)
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

  property("Backward Sampling is length filtered and contains the same times") {
    forAll (params) { p =>
      val data = observations(p)
      val filtered = KalmanFilter.kalmanFilter(mod, data, p)
      val sampled = Smoothing.backwardSampling(mod, filtered, p)

      assert(sampled.size === filtered.size)
      assert(sampled.map(_._1) === filtered.map(_.time))
    }
  }
}
