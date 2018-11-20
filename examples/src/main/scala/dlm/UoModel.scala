package examples.dlm

import dlm.core.model._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions._
import cats.implicits._
import scala.util.control.Exception._
import kantan.csv._
import kantan.csv.ops._
import kantan.csv.generic._
import java.time._
import java.time.format._
import java.time.temporal.ChronoUnit
import kantan.csv.java8._
import java.nio.file.Paths
import akka.actor.ActorSystem
import akka.Done
import akka.stream._
import scaladsl._
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

// Write this up in DLM Chapter
// Could additionally use Wishart for full rank Observation Matrix
trait RoundedUoData {
  val tempDlm = Dlm.polynomial(1) |+| Dlm.seasonal(24, 3)
  val dlmComp = List.fill(4)(tempDlm).reduce(_ |*| _)

  case class EnvSensor(
    sensorId:   String,
    datetime:    LocalDateTime,
    co:          Option[Double],
    humidity:    Option[Double],
    no:          Option[Double],
    temperature: Option[Double])

  val format = DateTimeFormatter.ISO_DATE_TIME
  def toDatetimeOpt(s: String, format: DateTimeFormatter) =
    catching(classOf[DateTimeException]) opt LocalDateTime.parse(s, format)

  def toDoubleOpt(s: String) =
    catching(classOf[NumberFormatException]) opt s.toDouble

  def parseEnvSensor(l: Vector[String]): Option[EnvSensor] =
    for {
      datetime <- toDatetimeOpt(l(1), format)
    } yield EnvSensor(l.head, datetime, toDoubleOpt(l(2)),
      toDoubleOpt(l(3)), toDoubleOpt(l(4)), toDoubleOpt(l(5)))

  val data = Streaming.readCsv("examples/data/summarised_sensor_data.csv").
    map(parseEnvSensor).
    collect { case Some(a) => a }

  val encodedData = Streaming.readCsv("examples/data/encoded_sensor_data.csv").
    map(parseEnvSensor).
    collect { case Some(a) => a }

  def envToData(a: EnvSensor): Data =
    Data(a.datetime.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0),
      DenseVector(a.co, a.humidity, a.no map math.log, a.temperature))
}

// fit nine independent models
object FitUoDlms extends App with EmoteData {
  implicit val system = ActorSystem("uo-dlm")
  implicit val materializer = ActorMaterializer()

  val priorV = InverseGamma(2.0, 3.0)
  val priorW = InverseGamma(2.0, 3.0)

  val dlmP = for {
    w <- priorW
    v <- priorV
    seasonalP = DlmParameters(
      v = DenseMatrix(v),
      w = diag(DenseVector.fill(7)(w)),
      m0 = DenseVector.zeros[Double](7),
      c0 = DenseMatrix.eye[Double](7))
  } yield List.fill(4)(seasonalP).reduce(Dlm.outerSumParameters)

  def writeSensor(id: String): Future[Done] = {
    val d = data.
      filter(_.sensorId == id).
      map(envToData)

    val iters = GibbsSampling.sample(dlmComp, priorV, priorW, dlmP.draw, d.toVector)
    Streaming
      .writeParallelChain(iters, 2, 10000, s"examples/data/uo_dlm_seasonal_daily_log_no_${id}",
        (s: GibbsSampling.State) => s.p.toList).
    runWith(Sink.ignore)
  }

  val ids = List("new_new_emote_1171", "new_new_emote_1172", "new_new_emote_1702",
              "new_new_emote_1708", "new_new_emote_2602", "new_new_emote_2603",
              "new_new_emote_2604", "new_new_emote_2605", "new_new_emote_2606")

  ids.map(writeSensor).sequence.
    onComplete { s =>
      println(s)
      system.terminate()
    }
}

object ForecastUoDlm extends App with RoundedUoData {
  implicit val system = ActorSystem("forecast_uo")
  implicit val materializer = ActorMaterializer()

  def envToDataTime(a: EnvSensor): (LocalDateTime, Data) =
    (a.datetime, Data(a.datetime.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0),
      DenseVector(a.co, a.humidity, a.no, a.temperature)))

  encodedData.
    filter(_.sensorId != "new_new_emote_1702").
    filter(_.sensorId != "new_new_emote_1108").
    groupBy(10000, _.sensorId).
    fold(("", Vector.empty[(LocalDateTime, Data)]))((l, r) =>
      (r.sensorId, l._2 :+ envToDataTime(r))).

    mergeSubstreams.
    mapAsyncUnordered(1) { case (id: String, d: Vector[(LocalDateTime, Data)]) =>
      val data = d.map(_._2)
      val times = d.map(_._1)

      println(s"Forecasting sensor $id")

      val file = s"examples/data/uo_dlm_seasonal_daily_${id}_0.csv"

      val ps = Streaming.readCsv(file).
        map(_.map(_.toDouble).toList).
        drop(1000).
        map(x => DlmParameters.fromList(4, 28)(x)).
        via(Streaming.meanParameters(4, 28)).
        runWith(Sink.head)
 
      val out = new java.io.File(s"examples/data/forecast_urbanobservatory_${id}.csv")
      val headers = rfc.withHeader(false)

      for {
        p <- ps
        filtered = KalmanFilter.filterDlm(dlmComp, data, p)
        summary = filtered.flatMap(kf => (for {
          f <- kf.ft
          q <- kf.qt
        } yield Dlm.summariseForecast(0.75)(f, q)).
          map(f => f.flatten.toList))
        toWrite: Vector[(String, LocalDateTime, List[Double])] = times.
          zip(summary).
          map(d => (id, d._1, d._2))
        io = out.writeCsv(toWrite, headers)
      } yield io
    }.
    runWith(Sink.onComplete { s =>
      println(s)
      system.terminate()
    })
}
