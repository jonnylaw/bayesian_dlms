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
import akka.actor.ActorSystem
import akka.stream._
import scaladsl._

trait NoData {
  val format = DateTimeFormatter.ISO_DATE_TIME
  implicit val codec: CellCodec[LocalDateTime] = localDateTimeCodec(format)

  def secondsToHours(seconds: Long): Double = {
    seconds / (60.0 * 60.0)
  }

  val rawData = Paths.get("examples/data/training_no.csv")
  val reader = rawData.asCsvReader[(LocalDateTime, Double)](rfc.withHeader)
  val data = reader
    .collect {
      case Right(a) =>
        Dlm.Data(secondsToHours(a._1.toEpochSecond(ZoneOffset.UTC)),
          DenseVector(Some(a._2)))
    }
    .toStream
    .zipWithIndex
    .filter { case (_, i) => i % 10 == 0 }
    .map(_._1)
    .toVector
    .take(5000)
}

/**
  * First order Students-t Distributed Model
  */
object NoStudentTModel extends App with NoData {
  implicit val system = ActorSystem("no-student-t-gibbs")
  implicit val materializer = ActorMaterializer()

  val dlm = Dlm.polynomial(1) |+| Dlm.seasonal(24, 3) |+| Dlm.seasonal(24 * 7, 3)
  val mod = Dglm.studentT(3, dlm)
  val params = DlmParameters(
    DenseMatrix(3.0),
    diag(DenseVector.fill(13)(0.1)),
    DenseVector.fill(13)(0.0),
    diag(DenseVector.fill(13)(100.0)))

  val priorW = InverseGamma(3.0, 3.0)
  val priorNu = Poisson(3)

  // nu is the mean of the negative binomial proposal (A Gamma mixture of Poissons)
  val propNu = (size: Double) => (nu: Int) => {
    val prob = nu / (size + nu)

    for {
      lambda <- Gamma(size, prob / (1 - prob))
      x <- Poisson(lambda)
    } yield x + 1
  }

  val propNuP = (size: Double) => (from: Int, to: Int) => {
    val r = size
    val p = from / (r + from)
    NegativeBinomial(p, r).logProbabilityOf(to)
  }

  val iters = StudentT
    .sample(data.toVector, priorW, priorNu, propNu(0.5), propNuP(0.5), mod, params)

  def formatParameters(s: StudentT.State) = {
    s.nu.toDouble :: DenseVector.vertcat(diag(s.p.v), diag(s.p.w)).data.toList
  }

  Streaming
    .writeParallelChain(iters, 2, 100000,
      "examples/data/no_dglm_seasonal_weekly_student_exact", formatParameters)
    .runWith(Sink.onComplete(_ => system.terminate()))
}

object NoGaussianModel extends App with NoData {
  implicit val system = ActorSystem("no-gaussian-gibbs")
  implicit val materializer = ActorMaterializer()

  val dlm = Dlm.polynomial(1) |+| Dlm.seasonal(24, 3) |+| Dlm.seasonal(24 * 7, 3)
  val params = DlmParameters(
    DenseMatrix(3.0),
    diag(DenseVector.fill(13)(0.1)),
    DenseVector.fill(13)(0.0),
    diag(DenseVector.fill(13)(100.0)))

  val priorW = InverseGamma(3.0, 3.0)
  val priorV = InverseGamma(3.0, 3.0)

  val iters = GibbsSampling.sampleSvd(dlm, priorV, priorW, params, data.toVector)

  def formatParameters(s: GibbsSampling.State) = {
    DenseVector.vertcat(diag(s.p.v), diag(s.p.w)).data.toList
  }

  Streaming
    .writeParallelChain(iters, 2, 100000, "examples/data/no_dlm_seasonal_weekly", formatParameters)
    .runWith(Sink.onComplete(_ => system.terminate()))
}

// TODO: Forecasting
