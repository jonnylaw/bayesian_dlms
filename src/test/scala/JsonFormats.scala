import org.scalacheck._
import Prop.forAll
import spray.json._
import JsonFormats._
import breeze.linalg.{DenseMatrix, DenseVector}
import Dlm._
import cats._
import cats.data._
import cats.implicits._
import Arbitrary.arbitrary

object JsonSpecification extends Properties("JsonFormat") {
  implicit def vectoreq = new Eq[DenseVector[Double]] {
    def eqv(x: DenseVector[Double], y: DenseVector[Double]) = {
      val tol = 1e-6
      x.data.zip(y.data).
        forall { case (a, b) => math.abs(a - b) < tol }
    }
  }

  implicit def dataeq(implicit dveq: DenseVector[Double]) = new Eq[Data] {
    def eqv(x: Data, y: Data) = (x.observation, y.observation) match {
      case (Some(a), Some(b)) =>
        math.abs(x.time - y.time) < 1e-6 & implicitly[Eq[DenseVector[Double]]].eqv(a, b)
      case (None, None) =>
        math.abs(x.time - y.time) < 1e-6
      case _ => false
    }
  }

  def denseVector = (n: Int) => Gen.containerOfN[Array, Double](n, arbitrary[Double]).
    map(a => DenseVector(a))

  implicit val dv = Arbitrary(denseVector(10))

  property("denseVectorJson") = forAll { (a: DenseVector[Double]) =>
    a.toJson.compactPrint.parseJson.convertTo[DenseVector[Double]] === a
  }

  def dataJson = for {
    t <- arbitrary[Int]
    y <- denseVector(10)
    obs <- Gen.oneOf(Some(y), None)
  } yield Data(t, obs)

  property("dataJson") = forAll { (a: Data) =>
    a.toJson.compactPrint.parseJson.convertTo[Data] === a
  }
}
