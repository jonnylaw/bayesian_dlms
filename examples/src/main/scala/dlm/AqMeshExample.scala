package examples.dlm

import dlm.core.model._
import breeze.linalg.{DenseVector, DenseMatrix, diag}
import breeze.stats.distributions._
import kantan.csv._
import kantan.csv.ops._
import kantan.csv.generic._
import java.time._
import java.time.format._
import kantan.csv.java8._
import java.nio.file.Paths
import akka.actor.ActorSystem
import akka.stream._
import scaladsl._

trait AqmeshModel {
  val format = DateTimeFormatter.ISO_DATE_TIME
  implicit val codec: CellCodec[LocalDateTime] =
    localDateTimeCodec(format)

  val seasonalDlm = Dlm.polynomial(1) |+| Dlm.seasonal(12, 3)
  val dlmComp = List.fill(3)(seasonalDlm).reduce(_ |*| _)

  def millisToHours(dateTimeMillis: Long) =
    dateTimeMillis / 1e3 * 60 * 60

  val priorBeta = Gaussian(0.0, 3.0)
  val priorSigmaEta = InverseGamma(2.5, 3.0)
  val priorPhi = Gaussian(0.8, 0.1)
  val priorSigma = InverseGamma(2.5, 3.0)
  val priorW = InverseGamma(2.5, 3.0)
  val priorMu = Gaussian(0.0, 1.0)

  val volP = for {
    se <- priorSigmaEta
    phi <- priorPhi
    mu <- priorMu
  } yield SvParameters(phi, mu, se)

  // two factors for the 21 dimensional state
  val fsv = for {
    bij <- priorBeta
    sigmaX <- priorSigma
    vp <- volP
  } yield FsvParameters(
    DenseMatrix.eye[Double](3) * sigmaX,
    FactorSv.buildBeta(21, 2, bij),
    Vector.fill(2)(vp))

  // the DLM has a 21 dimensional latent state (represented by 3 * 7 models)
  val initP = for {
    fs <- fsv
    v <- InverseGamma(3.0, 0.5)
    m0 = DenseVector.rand(21, Gaussian(0.0, 1.0))
    c0 = DenseMatrix.eye[Double](21)
    dlm = DlmParameters(
      diag(DenseVector.fill(3)(v)),
      DenseMatrix.eye[Double](21), m0, c0),
  } yield DlmFsvParameters(dlm, fs)

  case class AqmeshSensor(
    datetime:      LocalDateTime,
    humidity:      Option[Double],
    no:            Option[Double],
    no2:           Option[Double],
    o3:            Option[Double],
    particleCount: Option[Double],
    pm1:           Option[Double],
    pm10:          Option[Double],
    pm25:          Option[Double],
    pressure:      Option[Double],
    temperature:   Option[Double])

  val rawData = Paths.get("examples/data/aq_mesh_wide.csv")
  val reader = rawData.asCsvReader[AqmeshSensor](rfc.withHeader)
}

object FitAqMeshFull extends App with AqmeshModel {
  implicit val system = ActorSystem("aqmesh")
  implicit val materializer = ActorMaterializer()

  override val dlmComp = List.fill(7)(seasonalDlm).
    reduce(_ |*| _)

  override val fsv = for {
    bij <- priorBeta
    sigmaX <- priorSigma
    vp <- volP
  } yield FsvParameters(
    DenseMatrix.eye[Double](3) * sigmaX,
    FactorSv.buildBeta(49, 2, bij),
    Vector.fill(2)(vp))

  override val initP = for {
    fs <- fsv
    v <- InverseGamma(3.0, 0.5)
    m0 = DenseVector.rand(49, Gaussian(0.0, 1.0))
    c0 = DenseMatrix.eye[Double](49)
    dlm = DlmParameters(
      diag(DenseVector.fill(3)(v)),
      DenseMatrix.eye[Double](49), m0, c0),
  } yield DlmFsvParameters(dlm, fs)

  val training = reader.
    collect { case Right(a) => a }.
    filter(_.datetime.compareTo(
      LocalDateTime.of(2018, Month.JANUARY, 1, 0, 0)) > 0).
    filter(_.datetime.compareTo(
      LocalDateTime.of(2018, Month.FEBRUARY, 1, 0, 0)) < 0).
    toVector.
    zipWithIndex.
    filter { case (_, i) => i % 4 == 0 }. // thinned
    map(_._1).
    map(a => Data(
      a.datetime.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0),
      DenseVector(a.humidity, a.no, a.no2, a.o3,
        a.pm10, a.pm25, a.temperature))
    ).
    sortBy(_.time)

  val initialParams = initP.draw

  val iters = DlmFsvSystem.sample(priorBeta, priorSigmaEta,
    priorPhi, priorMu, priorSigma, priorW, training,
    dlmComp, initialParams)

  def formatParameters(s: DlmFsvSystem.State) =
    s.p.toList

  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/aqmesh_gibbs_full", formatParameters).
    runWith(Sink.onComplete { s =>
      println(s)
      system.terminate()
    })
}

object FitAqMesh extends App with AqmeshModel {
  implicit val system = ActorSystem("aqmesh")
  implicit val materializer = ActorMaterializer()

  val training = reader.
    collect { case Right(a) => a }.
    filter(_.datetime.compareTo(
      LocalDateTime.of(2018, Month.JANUARY, 1, 0, 0)) > 0).
    filter(_.datetime.compareTo(
      LocalDateTime.of(2018, Month.FEBRUARY, 1, 0, 0)) < 0).
    toVector.
    zipWithIndex.
    filter { case (_, i) => i % 4 == 0 }. // thinned
    map(_._1).
    map(a => Data(
      a.datetime.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0),
      DenseVector(a.no, a.no2, a.o3))
    ).
    sortBy(_.time)

  val initialParams = initP.draw

  val iters = DlmFsvSystem.sample(priorBeta, priorSigmaEta,
    priorPhi, priorMu, priorSigma, priorW, training,
    dlmComp, initialParams)

  def formatParameters(s: DlmFsvSystem.State) =
    s.p.toList

  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/aqmesh_params", formatParameters).
    runWith(Sink.onComplete { _ => system.terminate() })
}

object OneStepForecastAqmesh extends App with AqmeshModel {
  implicit val system = ActorSystem("aqmesh")
  implicit val materializer = ActorMaterializer()

  val data = reader.
    collect { case Right(a) => a }.
    toVector

  val training = data.
    filter(_.datetime.compareTo(
      LocalDateTime.of(2018, Month.JANUARY, 1, 0, 0)) > 0).
    filter(_.datetime.compareTo(
      LocalDateTime.of(2018, Month.FEBRUARY, 1, 0, 0)) < 0).
    zipWithIndex.
    filter { case (_, i) => i % 4 == 0 }. // thinned
    map(_._1).
    map(a => Data(
      a.datetime.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0),
      DenseVector(a.no, a.no2, a.o3))
    ).
    sortBy(_.time)

  val test = data.
    filter(_.datetime.compareTo(
      LocalDateTime.of(2018, Month.FEBRUARY, 1, 0, 0)) > 0).
    filter(_.datetime.compareTo(
      LocalDateTime.of(2018, Month.FEBRUARY, 7, 0, 0)) < 0).
    zipWithIndex.
    filter { case (_, i) => i % 4 == 0 }. // thinned
    map(_._1).
    map(a => Data(
      a.datetime.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0),
      DenseVector(a.no, a.no2, a.o3))
    ).
    sortBy(_.time)

  Streaming.
    readCsv("examples/data/aqmesh_gibbs_no_no2_o3_0.csv").
    drop(10000).
    map(_.map(_.toDouble).toList).
    map(DlmFsvSystem.paramsFromList(3, 21, 2)).
    via(Streaming.meanDlmFsvSystemParameters(3, 21, 2)).
    map { params =>

      val out = new java.io.File("examples/data/forecast_aqmesh.csv")
      val headers = rfc.withHeader(false)

      val forecast = DlmFsvSystem.forecast(dlmComp, params, test)

      val summary = forecast.flatMap(x => (for {
        f <- x.ft
        q <- x.qt
      } yield Dlm.summariseForecast(0.75)(f, q)).
        map(f => f.flatten.toList))

      out.writeCsv(summary, headers)
    }.
    runWith(Sink.onComplete{ s =>
      println(s)
      system.terminate()
    })
}
