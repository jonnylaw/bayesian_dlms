package examples.dlm

import dlm.core.model._
import breeze.stats.distributions._
import breeze.linalg.{DenseVector, DenseMatrix, diag}
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
import scala.concurrent.ExecutionContext.Implicits.global

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

  val seasonalDlm = Dlm.polynomial(1) |+| Dlm.seasonal(24, 3)
  val mvDlm = List.fill(8)(seasonalDlm).reduce(_ |*| _)
  val mvDlmShareState = seasonalDlm.
    copy(f = (t: Double) => List.fill(8)(seasonalDlm.f(t)).
           reduce((a, b) => DenseMatrix.horzcat(a, b)))

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
}

object TemperatureDlm extends App with TemperatureModel {
  implicit val system = ActorSystem("temperature_dlm")
  implicit val materializer = ActorMaterializer()

  val priorV = InverseGamma(3.0, 3.0)
  val priorW = InverseGamma(3.0, 3.0)

  val prior = for {
    v <- Applicative[Rand].replicateA(8, priorV)
    w <- Applicative[Rand].replicateA(7, priorW)
    m0 = Array.fill(7)(0.0)
    c0 = Array.fill(7)(10.0)
  } yield DlmParameters(
      v = diag(DenseVector(v.toArray)),
      w = diag(DenseVector(w.toArray)),
      m0 = DenseVector(m0),
      c0 = diag(DenseVector(c0)))

  val iters = GibbsSampling.sample(mvDlmShareState, priorV,
                                   priorW, prior.draw, ys)

  def formatParameters(s: GibbsSampling.State) =
    s.p.toList

  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/temperature_dlm_share_state", formatParameters).
    runWith(Sink.onComplete { s =>
              println(s)
              system.terminate()
            })
}

object ForecastTemperatureDlm extends App with TemperatureModel {
  implicit val system = ActorSystem("forecast_temperature")
  implicit val materializer = ActorMaterializer()

  val file = s"examples/data/temperature_dlm_share_state_0.csv"

  val ps = Streaming.readCsv(file).
    map(_.map(_.toDouble).toList).
    drop(1000).
    map(x => DlmParameters.fromList(8, 7)(x)).
    via(Streaming.meanParameters(8, 7)).
    runWith(Sink.head)

  val times: Vector[LocalDateTime] = data.map(_.date)

  val out = new java.io.File(s"examples/data/forecast_temperature.csv")
  val headers = rfc.withHeader(false)

  (for {
    p <- ps
    filtered = KalmanFilter.filterDlm(mvDlmShareState, ys, p)
    summary = filtered.flatMap(kf => (for {
                                        f <- kf.ft
                                        q <- kf.qt
                                      } yield Dlm.summariseForecast(0.75)(f, q)).
                                 map(f => f.flatten.toList))
    toWrite: Vector[(LocalDateTime, List[Double])] = times.
    zip(summary).
    map(d => (d._1, d._2))
    io = out.writeCsv(toWrite, headers)
  } yield io).
    onComplete(_ => system.terminate())
}

object StateTemperatureDlm extends App with TemperatureModel {
  implicit val system = ActorSystem("forecast_temperature")
  implicit val materializer = ActorMaterializer()

  val file = s"examples/data/temperature_dlm_share_state_0.csv"

  val ps = Streaming.readCsv(file).
    map(_.map(_.toDouble).toList).
    drop(1000).
    map(x => DlmParameters.fromList(8, 7)(x)).
    via(Streaming.meanParameters(8, 7)).
    runWith(Sink.head)

  val out = new java.io.File(s"examples/data/residuals_temperature_dlm.csv")
  val sensorNames = data.map(_.name).distinct.toList
  val headers = rfc.withHeader("day" :: sensorNames: _*)

  val res = for {
    p <- ps
    state = Smoothing.ffbsDlm(mvDlmShareState, ys, p).sample(1000).
      map(_.map(x => x.copy(sample = mvDlmShareState.f(x.time).t * x.sample)))
    times: Vector[LocalDateTime] = data.map(_.date)
    meanState = state.transpose.zip(times).
      map { case (x, t) => (t, x.map(_.sample).reduce(_ + _).map(_ / 1000).data.toList) }.toList
    io = out.writeCsv(meanState, headers)
  } yield io

  res.onComplete{ s =>
    println(s)
    system.terminate()
  }
}
