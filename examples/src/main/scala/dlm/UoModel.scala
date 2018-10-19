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

trait JointUoModel {
  val format = DateTimeFormatter.ISO_DATE_TIME
  implicit val codec: CellCodec[LocalDateTime] =
    localDateTimeCodec(format)
  val tempDlm = Dlm.polynomial(1) |+| Dlm.seasonal(24, 3)
  val foDlm = Dlm.polynomial(1)
  val dlmComp = List(foDlm, tempDlm, foDlm, tempDlm).
    reduce(_ |*| _)

  def millisToHours(dateTimeMillis: Long) = {
    dateTimeMillis / 1e3 * 60 * 60
  }

  val priorBeta = Gaussian(0.0, 3.0)
  val priorSigmaEta = InverseGamma(2.5, 3.0)
  val priorPhi = new Beta(20, 2)
  val priorSigma = InverseGamma(2.5, 3.0)
  val priorW = InverseGamma(2.5, 3.0)
  val priorMu = Gaussian(0.0, 1.0)

  val volP = for {
    se <- priorSigmaEta
    phi <- priorPhi
    mu <- priorMu
  } yield SvParameters(phi, mu, se)

  val fsvP = for {
    bij <- priorBeta
    sigma <- priorSigma
    vp <- volP
  } yield FsvParameters(sigma,
    FactorSv.buildBeta(4, 2, bij),
    Vector.fill(2)(vp))

  val dlmP = for {
    w <- priorW
    foP = DlmParameters(
      v = DenseMatrix.eye[Double](1),
      w = DenseMatrix(w),
      m0 = DenseVector.zeros[Double](1),
      c0 = DenseMatrix.eye[Double](1))
    seasonalP = DlmParameters(
      v = DenseMatrix.eye[Double](1),
      w = diag(DenseVector.fill(7)(w)),
      m0 = DenseVector.zeros[Double](7),
      c0 = DenseMatrix.eye[Double](7))
  } yield List(foP, seasonalP, foP, seasonalP).
    reduce(Dlm.outerSumParameters)

  val initP: Rand[DlmFsvParameters] = for {
    fsv <- fsvP
    dlm <- dlmP
  } yield DlmFsvParameters(dlm, fsv) 

  case class EnvSensor(
    datetime:    LocalDateTime,
    co:          Option[Double],
    humidity:    Option[Double],
    no:          Option[Double],
    temperature: Option[Double])

  val rawData = Paths.get("examples/data/new_new_emote_1108_wide.csv")
  val reader = rawData.asCsvReader[EnvSensor](rfc.withHeader)
  val data = reader.
    collect {
      case Right(a) => a
    }.
    filter(_.datetime.compareTo(
      LocalDateTime.of(2017, Month.SEPTEMBER, 1, 0, 0)) > 0).
    filter(_.datetime.compareTo(
      LocalDateTime.of(2017, Month.OCTOBER, 1, 0, 0)) < 0).
    // time in hours
    map(a => Data(a.datetime.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0), 
        DenseVector(a.co, a.humidity, a.no, a.temperature))).
    toVector.
    sortBy(_.time).
    zipWithIndex.
    filter { case (_, i) => i % 10 == 0 }. // thinned
    map(_._1)
}

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
      DenseVector(a.co, a.humidity, a.no, a.temperature))
}

// fit ten independent models
object FitUoDlms extends App with RoundedUoData {
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

  def writeSensor(id: String): Future[Done] = for {
    d <- data.
      filter(_.sensorId == id).
      map(envToData).
      runWith(Sink.seq)
    iters = GibbsSampling.sample(dlmComp, priorV, priorW, dlmP.draw, d.toVector)
    io <- Streaming
      .writeParallelChain(iters, 2, 10000, s"examples/data/uo_dlm_seasonal_daily_${id}",
        (s: GibbsSampling.State) => DlmParameters.toList(s.p)).
    runWith(Sink.ignore)
  } yield io

  val ids = List("new_new_emote_1171", "new_new_emote_1172", "new_new_emote_1702",
    "new_new_emote_1708", "new_new_emote_2602", "new_new_emote_2603",
    "new_new_emote_2604", "new_new_emote_2605", "new_new_emote_2606",
    "new_new_emote_1108")

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
    filter(_.sensorId == "new_new_emote_1171").
    groupBy(10000, _.sensorId).
    fold(("Hello", Vector.empty[(LocalDateTime, Data)]))((l, r) =>
      (r.sensorId, l._2 :+ envToDataTime(r))).
    mergeSubstreams.
    mapAsyncUnordered(1) { case (id: String, d: Vector[(LocalDateTime, Data)]) =>
      val data = d.map(_._2)
      val times = d.map(_._1)

      val file = s"examples/data/uo_dlm_seasonal_daily_${id}_0.csv"

      val ps: Future[DlmParameters] = Streaming.readCsv(file).
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

object FitUoDlmStudent extends App with JointUoModel {

}

object FitContUo extends App with JointUoModel {
  implicit val system = ActorSystem("dlm_fsv_uo")
  implicit val materializer = ActorMaterializer()

  val iters = DlmFsv.sampleOu(priorBeta, priorSigmaEta,
    priorPhi, priorMu, priorSigma, priorW,
    data, dlmComp, initP.draw)

  def diagonal(m: DenseMatrix[Double]): List[Double] =
    for {
      i <- List.range(0, m.cols)
    } yield m(i,i)

  def format(s: DlmFsv.State): List[Double] =
    s.p.toList

  // write iterations
  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/uo_cont_gibbs_two_factors", format).
    runWith(Sink.onComplete(_ => system.terminate()))
}

object InterpolateUo extends App with JointUoModel {
  def readValue(a: String): Option[Double] = 
    if (a == "NA") None else Some(a.toDouble)

  val file = "examples/data/uo_cont_gibbs_two_factors_0.csv"

  // read in parameters and calculate the mean
  val rawParams = Paths.get(file)
  val reader2 = rawParams.asCsvReader[List[Double]](rfc.withHeader)
  val params: DlmFsvParameters = reader2.
    collect { case Right(a) => a }.
    map(x => DlmFsvParameters.fromList(4, 16, 4, 2)(x)).
    foldLeft((DlmFsvParameters.empty(4, 16, 4, 2), 1.0)){
        case ((avg, n), b) =>
          (avg.map(_ * n).add(b).map(_  / (n + 1)), n + 1)
      }._1

  // read in data with encoded missing bits
  val rawData1 = Paths.get("examples/data/new_new_emote_1108_rounded.csv")
  val reader1 = rawData1.asCsvReader[(LocalDateTime, String,
    String, String, String)](rfc.withHeader)
  val testData = reader1.
    collect {
      case Right(a) => Data(a._1.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0),
        DenseVector(readValue(a._2), readValue(a._3),
          readValue(a._4), readValue(a._5)))
    }.
    toVector.
    sortBy(_.time)

  // use mcmc to sample the missing observations
  val iters = DlmFsv.sampleStateOu(testData, dlmComp, params).
    steps.
    take(1000).
    map { (s: DlmFsv.State) =>
      val vol = s.volatility.map(x => (x.time, x.sample))
      val st = s.theta.map(x => (x.time, x.sample))
      DlmFsv.obsVolatility(vol, st, dlmComp, params)
    }.
    map(s => s.map { case (t, y) => (t, Some(y(0))) }).
    toVector

  // calculate credible intervals
  val summary = DlmFsv.summariseInterpolation(iters, 0.995)

  // write interpolated data
  val out = new java.io.File("examples/data/interpolated_urbanobservatory.csv")
  val headers = rfc.withHeader("time", "mean", "upper", "lower")

  out.writeCsv(summary, headers)
}

object ForecastUo extends App {
  implicit val system = ActorSystem("forecast-uo")
  implicit val materializer = ActorMaterializer()

  val dlm = Dlm.polynomial(1) |+|
    Dlm.seasonal(24, 3) |+| Dlm.seasonal(24 * 7, 3)
  val mod = Dglm.studentT(3, dlm)

  val file = "examples/data/uo_cont_gibbs_two_factors_0.csv"

  // read parameters from file
  val ps: Future[DlmFsvParameters] = Streaming.readCsv(file).
    map(_.map(_.toDouble).toList).
    map(x => DlmFsvParameters.fromList(4, 16, 4, 2)(x)).
    via(Streaming.meanDlmFsvParameters(4, 16, 4, 2)).
    runWith(Sink.head)

  val out = new java.io.File("examples/data/forecast_uo.csv")
  val headers = rfc.withHeader("time", "mean", "lower", "upper")

  val n = 1000
}
