import org.scalacheck._
import Prop.forAll
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import cats._
import cats.data._
import cats.implicits._
import Arbitrary.arbitrary
import math.exp
import dlm.model._
import KalmanFilter._

object KfSpecification extends Properties("KalmanFilter") {

}
