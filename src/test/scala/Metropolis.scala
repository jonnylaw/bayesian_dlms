import dlm.model._
import breeze.linalg.{DenseMatrix, DenseVector, cond, diag}
import breeze.stats.distributions.{ChiSquared, Gamma}
import breeze.stats.covmat
import breeze.stats.{meanAndVariance, variance, mean}
import org.scalatest._
import prop._
import org.scalacheck.Gen
import org.scalactic.Equality

class MetropolisHastings extends PropSpec with GeneratorDrivenPropertyChecks with Matchers with BreezeGenerators {
  def diagonalMatrix(n: Int) = 
    for {
      a <- smallDouble
    } yield diag(DenseVector.fill(n)(a))


  property("Propose Matrix Should be positive definite") {
    forAll(diagonalMatrix(2)) { (m: DenseMatrix[Double]) =>
      val prop = MetropolisHastings.proposeDiagonalMatrix(0.05)(m).draw
      val diagonalValues = diag(m).data

      assert(m.cols === prop.cols)
      assert(diagonalValues.forall(a => a >= 0))
    }
  }
}
