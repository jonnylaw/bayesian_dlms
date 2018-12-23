import dlm.core.model._
import org.scalatest._
import org.scalactic.Equality
import breeze.linalg.{DenseVector, DenseMatrix}

class ParticleGibbsTest extends FunSuite with Matchers with BreezeGenerators {
  implicit val tol = 1e-2

  // simulate data
  val mod = Dglm.poisson(Dlm.polynomial(1))
  val params = DlmParameters(DenseMatrix(2.0),
                             DenseMatrix(0.05),
                             DenseVector(0.0),
                             DenseMatrix(1.0))

  val data =
    Dglm.simulateRegular(mod, params, 1.0).steps.take(10).toVector.map(_._1)
  val n = 200
  val pf = ParticleGibbs(n)

  test("Initial Conditioned State matches times") {
    val filtered = pf.initialiseState(mod, params, data)
    val conditionedState = filtered.conditionedState.map(_._1).toVector.sorted

    conditionedState foreach println

    assert(data.map(_.time) === conditionedState)
  }
}
