package examples.dlm

import dlm.core.model._
import breeze.stats.distributions._
import breeze.linalg.{DenseVector, diag}
import java.nio.file.Paths
import cats._
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import kantan.csv.generic._
import java.time._
import java.time.format._
import kantan.csv.java8._
import akka.actor.ActorSystem
import akka.stream._
import scaladsl._

trait TemperatureModel {
  val format = DateTimeFormatter.ISO_DATE_TIME
  implicit val codec: CellCodec[LocalDateTime] =
    localDateTimeCodec(format)

  case class Temperature(
    name:     String,
    date:     LocalDateTime,
    obs:      Option[Double],
    lon:      Double,
    lat:      Double)

  val rawData = Paths.get("examples/data/daily_average_temp.csv")
  val reader = rawData.asCsvReader[Temperature](rfc.withHeader)
  val data: Vector[Temperature] = reader.
    collect {
      case Right(a) => a
    }.
    toVector
}

object TemperatureDlm extends App with TemperatureModel {
  implicit val system = ActorSystem("temperature_dlm")
  implicit val materializer = ActorMaterializer()

  val seasonalDlm = Dlm.polynomial(1) |+| Dlm.seasonal(24, 3)
  val mvDlm = List.fill(8)(seasonalDlm).reduce(_ |*| _)

  val ys = data.
    groupBy(_.date).
    map { case (t, temps) =>
      Data(t.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0), // time in hours
           DenseVector(temps.map(s => s.obs).toArray))
    }.
    toVector.
    sortBy(_.time).
    zipWithIndex.
    map { case (d, i) => d.copy(time = i.toDouble) }

  val priorV = InverseGamma(3.0, 3.0)
  val priorW = InverseGamma(3.0, 3.0)

  val prior = for {
    v <- Applicative[Rand].replicateA(8, priorV)
    w <- Applicative[Rand].replicateA(56, priorW)
    m0 <- Applicative[Rand].replicateA(56, Gaussian(0.0, 1.0))
    c0 <- Applicative[Rand].replicateA(56, InverseGamma(3.0, 0.5))
  } yield DlmParameters(
      v = diag(DenseVector(v.toArray)),
      w = diag(DenseVector(w.toArray)),
      m0 = DenseVector(m0.toArray),
      c0 = diag(DenseVector(c0.toArray)))

  val iters = GibbsSampling.sample(mvDlm, priorV, priorW, prior.draw, ys)

  def formatParameters(s: GibbsSampling.State) =
    s.p.toList

  Streaming.writeParallelChain(
    iters, 2, 10000, "examples/data/temperature_dlm", formatParameters).
    runWith(Sink.onComplete { s =>
              println(s)
              system.terminate()
            })
}

