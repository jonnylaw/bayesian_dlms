import core.dlm.model._
import org.scalatest._
import org.scalactic.Equality
import breeze.linalg.{diag, DenseVector, DenseMatrix, svd, eigSym, inv}
import breeze.stats.distributions.MultivariateGaussian
import cats.instances.vector._
import cats.syntax.traverse._

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
    map { case (t, y, a) => Dlm.Data(t, DenseVector(y)) }

  val filteredTest = KalmanFilter.filter[Vector](model, sims, params, FilterAr.advanceState(p))
  val filterSvdTest = SvdFilter.filter(model, sims, params, FilterAr.advanceStateSvd(p))

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

  // calculate AR(1) filter "by hand" to test SVD Filter
  val data = Vector(
    Dlm.Data(1.0, DenseVector(Some(4.5), Some(4.5))),
    Dlm.Data(2.0, DenseVector(Some(3.0), Some(3.0))),
    Dlm.Data(3.0, DenseVector(Some(6.3), Some(6.3))),
    Dlm.Data(4.0, DenseVector[Option[Double]](None, None)),
    Dlm.Data(5.0, DenseVector(Some(10.1), None)), // partially observed
    Dlm.Data(7.0, DenseVector(Some(15.2), Some(15.2)))
  )


}
