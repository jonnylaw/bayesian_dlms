import org.scalacheck._
import Prop.forAll
import dlm.model._
import JsonFormats._
import Dlm._
import breeze.linalg.{DenseMatrix, DenseVector}
import cats._
import cats.data._
import cats.implicits._
import Arbitrary.arbitrary

object JsonSpecification extends Properties("JsonFormat") {
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

  implicit val arbData = Arbitrary(dataJson)

  property("dataJson") = forAll { (a: Data) =>
    a.toJson.compactPrint.parseJson.convertTo[Data] === a
  }
}
