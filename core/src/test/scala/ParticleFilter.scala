import dlm.core.model._
import breeze.linalg.{Vector => _, _}
import org.scalatest._
import prop._
import org.scalactic.Equality
import org.scalacheck.Gen

class ParticleFilter
    extends PropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers
    with BreezeGenerators {

  val params = for {
    v <- smallDouble
    w <- symmetricPosDefMatrix(2, 100)
    m0 = DenseVector.zeros[Double](2)
    c0 = DenseMatrix.eye[Double](2) * 100.0
  } yield DlmParameters(DenseMatrix(v), w, m0, c0)

  implicit def matrixeq(implicit tol: Double, eqm: Equality[DenseMatrix[Double]], eqv: Equality[DenseVector[Double]]) =
    new Equality[DlmParameters] {
      def areEqual(p: DlmParameters, b: Any) = b match {
        case DlmParameters(v, w, m, c) =>
          eqm.areEqual(p.v, v) & eqm.areEqual(p.w, w) & eqv.areEqual(p.m0, m) & eqm.areEqual(p.c0, c)
        case _ => false
      }
    }

  property("Mean of collection of parameters") {
    forAll(params) { p =>
      implicit val tol = 0.01
      val ps = Vector.fill(100)(p)
      val ws = Vector.fill(100)(1.0)
      val meanPs = LiuAndWestFilter.meanParameters(ps)

      assert(meanPs === p)
      assert(LiuAndWestFilter.weightedMeanParams(ps, ws) === meanPs)
    }
  }

  val n = 100
  val collectionParams = Gen.nonEmptyListOf(params)

  property("Variance of collection of parameters") {
    forAll(collectionParams) { ps =>
      implicit val tol = 0.01
      val ws = Vector.fill(n)(1.0)
      val variance = LiuAndWestFilter.weightedVarParameters(ps.toVector, ws)
      val variance2 = LiuAndWestFilter.varParameters(ps.toVector)

      assert(variance === variance2)
    }
  }
}
