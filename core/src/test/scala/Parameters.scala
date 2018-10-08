import dlm.core.model._
import org.scalatest._
import breeze.linalg.diag
import prop._
import org.scalactic.Equality
import org.scalacheck.Gen

class ReadingParameters
    extends PropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers
    with BreezeGenerators {

  def dlmParameters(vDim: Int, wDim: Int) = for {
    v <- denseVector(vDim)
    w <- denseVector(wDim)
    m0 <- denseVector(wDim)
    c0 <- denseVector(wDim)
  } yield DlmParameters(diag(v), diag(w), m0, diag(c0))

  property("Observation: toList for DLM parameters is inverse of fromList") {
    forAll(dlmParameters(2, 2)) { ps =>
      assert(DlmParameters.fromList(2, 2)(ps.toList).v === ps.v)
    }
  }

  property("System: toList for DLM parameters is inverse of fromList") {
    forAll(dlmParameters(2, 2)) { ps =>
      assert(DlmParameters.fromList(2, 2)(ps.toList).w === ps.w)
    }
  }

  property("Init mean: toList for DLM parameters is inverse of fromList") {
    forAll(dlmParameters(2, 2)) { ps =>
      assert(DlmParameters.fromList(2, 2)(ps.toList).m0 === ps.m0)
    }
  }

  property("Init Covariance: toList for DLM parameters is inverse of fromList") {
    forAll(dlmParameters(2, 2)) { ps =>
      assert(DlmParameters.fromList(2, 2)(ps.toList).c0 === ps.c0)
    }
  }
}
