import cats._
import cats.implicits._
import cats.laws.discipline._
import cats.kernel.laws._
import cats.kernel.instances.all._
import cats.kernel.laws.discipline._

import org.typelevel.discipline.Laws
import org.typelevel.discipline.scalatest.Discipline

import dlm.core.model._
import org.scalatest._
import cats.implicits._
import breeze.linalg.diag
import prop._
import org.scalactic.Equality
import org.scalacheck.Gen
import org.scalacheck._
import Arbitrary.arbitrary

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

  def svParameters = for {
    phi <- smallDouble
    mu <- smallDouble
    sigma <- smallDouble
  } yield SvParameters(phi, mu, sigma)

  property("Stochastic Volatility Parameters: toList is inverse of fromList") {
    forAll(svParameters) { ps =>
      assert(SvParameters.fromList(ps.toList) === ps)
    }
  }

  def factorParameters(p: Int, k: Int) = for {
    v <- smallDouble
    beta <- denseMatrix(p, k)
    fsv <- Gen.listOfN(k, svParameters)
  } yield FsvParameters(v, beta, fsv.toVector)

  property("Factor Stochastic Volatility Parameters: toList is inverse of fromList") {
    forAll(factorParameters(2, 2)) { ps =>
      assert(FsvParameters.fromList(2, 2)(ps.toList) === ps)
    }
  }

  def dlmFsvParameters(vDim: Int, wDim: Int, p: Int, k: Int) = for {
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

  def svParameters = for {
    phi <- smallDouble
    mu <- smallDouble
    sigma <- smallDouble
  } yield SvParameters(phi, mu, sigma)

  def factorParameters = for {
    v <- smallDouble
    p <- arbitrary[Int]
    k <- arbitrary[Int]
    beta <- denseMatrix(p, k)
    fsv <- Gen.listOfN(k, svParameters)
  } yield FsvParameters(v, beta, fsv.toVector)

  implicit def genFsvArb: Arbitrary[FsvParameters] =
    Arbitrary(factorParameters)

  implicit def eqFsv = new Eq[FsvParameters] {
    def eqv(x: FsvParameters, y: FsvParameters) = {
      x.v === y.v && x.beta === y.beta &&
        x.factorParams.zip(y.factorParams).
          forall { case (xi, yi) => xi === yi }
    }
  }

  checkAll("Additive Fsv Parameter Semigroup",
           SemigroupTests[FsvParameters].semigroup)
}
