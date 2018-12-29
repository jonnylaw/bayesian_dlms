import cats._
import cats.implicits._
import cats.laws.discipline._
import cats.kernel.instances.all._
import cats.kernel.laws.discipline._

import org.typelevel.discipline.scalatest.Discipline

import com.github.jonnylaw.dlm._
import org.scalatest._
import cats.implicits._
import breeze.linalg.diag
import prop._
import org.scalacheck.Gen
import org.scalacheck._
import Arbitrary.arbitrary

class ReadingParameters
    extends PropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers
    with BreezeGenerators {

  def dlmParameters(vDim: Int, wDim: Int) =
    for {
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

  def svParameters =
    for {
      phi <- smallDouble
      mu <- smallDouble
      sigma <- smallDouble
    } yield SvParameters(phi, mu, sigma)

  property("Stochastic Volatility Parameters: toList is inverse of fromList") {
    forAll(svParameters) { ps =>
      assert(SvParameters.fromList(ps.toList) === ps)
    }
  }

  def factorParameters(p: Int, k: Int) =
    for {
      v <- smallDouble
      beta <- denseMatrix(p, k)
      fsv <- Gen.listOfN(k, svParameters)
    } yield FsvParameters(v, beta, fsv.toVector)

  property(
    "Factor Stochastic Volatility Parameters: toList is inverse of fromList") {
    forAll(factorParameters(2, 2)) { ps =>
      assert(FsvParameters.fromList(2, 2)(ps.toList) === ps)
    }
  }

  def dlmFsvParameters(vDim: Int, wDim: Int, p: Int, k: Int) =
    for {
      dlm <- dlmParameters(vDim, wDim)
      fsv <- factorParameters(p, k)
    } yield DlmFsvParameters(dlm, fsv)

  property("DLM FSV Parameters: toList is inverse of fromList") {
    forAll(dlmFsvParameters(2, 2, 2, 2)) { ps =>
      assert(DlmFsvParameters.fromList(2, 2, 2, 2)(ps.toList) === ps)
    }
  }
}

class ParameterLaws
    extends FunSuite
    with Matchers
    with BreezeGenerators
    with Discipline {

  def svParameters: Gen[SvParameters] =
    for {
      phi <- smallDouble
      mu <- smallDouble
      sigma <- smallDouble
    } yield SvParameters(phi, mu, sigma)

  implicit val svArb = Arbitrary(svParameters)

  implicit def eqDouble(implicit tol: Int) = new Eq[Double] {
    def eqv(x: Double, y: Double): Boolean =
      math.abs(x - y) < math.pow(1, -tol)
  }

  implicit def eqSv(implicit eqd: Eq[Double]): Eq[SvParameters] = new Eq[SvParameters] {
    def eqv(x: SvParameters, y: SvParameters): Boolean = {
      eqd.eqv(x.phi, y.phi) & eqd.eqv(x.mu, y.mu) & eqd.eqv(x.sigmaEta, y.sigmaEta)
    }
  }

  implicit val tol = 2

  checkAll("Additive SV Parameter Semigroup",
           SemigroupTests[SvParameters].semigroup)
}
