// import org.scalacheck._
// import Prop.forAll
// import dlm.model._
// import JsonFormats._
// import Dlm._
// import breeze.linalg.{DenseMatrix, DenseVector}
// import cats._
// import cats.data._
// import cats.implicits._
// import Arbitrary.arbitrary
// import spray.json._

// object JsonSpecification extends Properties("JsonFormat") {
//   def denseVector = (n: Int) => Gen.containerOfN[Array, Double](n, arbitrary[Double]).
//     map(a => DenseVector(a))

  // implicit def dataeq = new Eq[Data] {
  //   def eqv(x: Data, y: Data) = (x.observation, y.observation) match {
  //     case (Some(a), Some(b)) =>
  //       x.time == y.time & implicitly[Eq[DenseVector[Double]]].eqv(a, b)
  //     case (None, None) =>
  //       x.time == y.time
  //     case _ => false
  //   }
  // }

//   implicit val dv = Arbitrary(denseVector(10))

//   property("denseVectorJson") = forAll { (a: DenseVector[Double]) =>
//     a.toJson.compactPrint.parseJson.convertTo[DenseVector[Double]] === a
//   }

//   def dataJson = for {
//     t <- arbitrary[Int]
//     y <- denseVector(10)
//     obs <- Gen.oneOf(Some(y), None)
//   } yield Data(t, obs)

//   implicit val arbData = Arbitrary(dataJson)

//   // property("dataJson") = forAll { (a: Data) =>
//   //   a.toJson.compactPrint.parseJson.convertTo[Data] === a
//   // }
// }
