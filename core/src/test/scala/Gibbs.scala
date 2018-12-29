import com.github.jonnylaw.dlm._
import org.scalatest._
import prop._
import org.scalactic.Equality

class GibbsTest
    extends PropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers
    with BreezeGenerators {}
