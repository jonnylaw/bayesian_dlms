import dlm.model._
import org.scalatest._
import prop._
import org.scalactic.Equality

class GibbsTest extends PropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers
    with BreezeGenerators {

  val w = symmetricPosDefMatrix(2, 100)
  val mod = Dlm.polynomial(2)
  val state = w map (Dlm.simulateStateRegular(mod, _).steps.take(100).toArray)
}
