package examples.dlm

import core.dlm.model._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions._
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import java.time._
import java.time.format._
import kantan.csv.java8._

/**
  * First order Students-t Distributed Model
  */
object NoStudentTModel extends App {
  val format = DateTimeFormatter.ISO_DATE_TIME
  implicit val codec: CellCodec[LocalDateTime] = localDateTimeCodec(format)

  val rawData = Paths.get("examples/data/training_no.csv")
  val reader = rawData.asCsvReader[(LocalDateTime, Double)](rfc.withHeader)
  val data = reader
    .collect {
      case Right(a) =>
        Dlm.Data(a._1.toEpochSecond(ZoneOffset.UTC), DenseVector(Some(a._2)))
    }
    .toStream
    .zipWithIndex
    .filter { case (_, i) => i % 30 == 0 }
    .map(_._1)
    .toVector
    .take(1000)

  val dlm = Dlm.polynomial(1)
  val mod = Dglm.studentT(3, dlm)
  val params = DlmParameters(DenseMatrix(3.0),
                             DenseMatrix(0.1),
                             DenseVector(0.0),
                             DenseMatrix(100.0))

  val priorW = InverseGamma(3.0, 3.0)
  val priorNu = Poisson(3)
  val propNu = (nu: Int) => Poisson(nu)

  val iters = StudentT
    .sample(data.toVector, priorW, priorNu, propNu, mod, params)
    .steps
    .take(10000)
    .map(_.p)

  def formatParameters(p: DlmParameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  val headers = rfc.withHeader("scale", "W")
  Streaming.writeChain(formatParameters,
                       "examples/data/no_dglm_exact.csv",
                       headers)(iters)
}
