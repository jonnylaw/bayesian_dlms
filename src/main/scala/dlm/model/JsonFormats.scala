package dlm.model

import spray.json._
import Dlm._
import KalmanFilter._
import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.distributions.MultivariateGaussian

object JsonFormats extends DefaultJsonProtocol {
  implicit def denseVectorFormat = new RootJsonFormat[DenseVector[Double]] {
    def write(vec: DenseVector[Double]) = JsArray(vec.data.map(_.toJson).toVector)

    def read(value: JsValue) = value match {
      case JsArray(elements) => {
        val s: Array[Double] = elements.map(_.convertTo[Double]).toArray[Double]
        DenseVector(s)
      }
      case JsNumber(a) => DenseVector(a.toDouble)
      case x => deserializationError("Expected DenseVector as JsArray, but got " + x)
    }
  }

  implicit object DataJsonFormat extends RootJsonFormat[Data] {
    def write(d: Data) = d.observation match {
      case Some(y: DenseVector[Double]) =>
        JsArray(JsNumber(d.time), JsArray(Vector(y.toJson)))
      case None => 
        JsArray(JsNumber(d.time), JsString("NA"))
    }

    def read(value: JsValue) = value match {
      case JsArray(Vector(JsNumber(t), JsString("NA"))) => 
        Data(t.toInt, None)
      case JsArray(Vector(JsNumber(t), y)) =>
        Data(t.toInt, Some(y.convertTo[DenseVector[Double]]))
      case _ => 
        deserializationError("Data expected")
    }
  }

  implicit def denseMatrixFormat = new RootJsonFormat[DenseMatrix[Double]] {
    def write(mat: DenseMatrix[Double]) = JsObject(
      "rows" -> JsNumber(mat.rows),
      "cols" -> JsNumber(mat.cols),
      "elements" -> JsArray(mat.data.map(_.toJson).toVector)
    )

    def read(value: JsValue) = {
      value.asJsObject.getFields("rows", "cols", "elements") match {
        case Seq(JsNumber(rows), JsNumber(cols), JsArray(elements)) =>
          val s: Array[Double] = elements.map(_.convertTo[Double]).toArray[Double]
          new DenseMatrix(rows.toInt, cols.toInt, s)
        case x => deserializationError("Expected DenseMatrix as JsArray, but got " + x)
      }
    }
  }

  implicit def stateFormat = jsonFormat2(MultivariateGaussian.apply)

  implicit object KalmanStateFormat extends RootJsonFormat[KfState] {
    def write(d: KfState) = d.y match {
      case Some(y) =>
        JsArray(
          JsNumber(d.time), d.statePosterior.toJson,
          d.statePrior.toJson, y.toJson, d.cov.get.toJson, JsNumber(d.ll)
        )
      case None => 
        JsArray(
          JsNumber(d.time), d.statePosterior.toJson, 
          d.statePrior.toJson, JsString("NA"), JsString("NA"), JsNumber(d.ll)
        )
    }

    def read(value: JsValue) = value match {
      case JsArray(Vector(JsNumber(t), statePost, statePrior, JsString("NA"), JsString("NA"), JsNumber(ll))) => 
        KfState(
          t.toInt, statePost.convertTo[State],
          statePrior.convertTo[State], None, None, ll.toDouble
        )
      case JsArray(Vector(JsNumber(t), statePost, statePrior, observation, cov, JsNumber(ll))) => 
        KfState(
          t.toInt, statePost.convertTo[State],
          statePrior.convertTo[State],
          Some(observation.convertTo[DenseVector[Double]]),
          Some(cov.convertTo[DenseMatrix[Double]]), 
          ll.toDouble)
      case _ => 
        deserializationError("KfState expected")
    }
  }
}
