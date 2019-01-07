package com.github.jonnylaw.dlm.example

import com.github.jonnylaw.dlm._
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
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

trait EmoteData {
  val format = DateTimeFormatter.ISO_DATE_TIME
  implicit val codec: CellCodec[LocalDateTime] =
    localDateTimeCodec(format)

  val tempDlm = Dlm.polynomial(1) |+| Dlm.seasonal(24, 3)
  val dlmComp = List.fill(4)(tempDlm).reduce(_ |*| _)

  case class EnvSensor(sensorId: String,
                       datetime: LocalDateTime,
                       co: Option[Double],
                       humidity: Option[Double],
                       no: Option[Double],
                       temperature: Option[Double])

  val rawData = Paths.get("examples/data/encoded_sensor_data.csv")
  val reader = rawData.asCsvReader[EnvSensor](rfc.withHeader)
  val data: Vector[EnvSensor] = reader.collect {
    case Right(a) => a
    case Left(ex) => throw new Exception(s"Truely bad things $ex")
  }.toVector

  val trainingData = data.filter(
    _.datetime.compareTo(LocalDateTime.of(2018, Month.JUNE, 1, 0, 0)) < 0)

  val testData = data
    .filter(
      _.datetime.compareTo(LocalDateTime.of(2018, Month.AUGUST, 1, 0, 0)) > 0)
    .filter(
      _.datetime.compareTo(LocalDateTime.of(2018, Month.SEPTEMBER, 1, 0, 0)) < 0)

  def envToData(a: EnvSensor): Data =
    Data(a.datetime.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0),
         DenseVector(a.co map math.log, a.humidity, a.no map math.log, a.temperature))
}

object FitFactorUo extends App with EmoteData {
  implicit val system = ActorSystem("dlm_fsv_uo")
  implicit val materializer = ActorMaterializer()

  val priorBeta = Gaussian(0.0, 3.0)
  val priorSigmaEta = InverseGamma(2.5, 3.0)
  val priorPhi = Gaussian(0.8, 0.1)
  val priorSigma = InverseGamma(2.5, 3.0)
  val priorV = InverseGamma(2.5, 3.0)
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
  } yield
    FsvParameters(DenseMatrix.eye[Double](28) * sigma,
                  FactorSv.buildBeta(28, 1, bij),
                  Vector.fill(1)(vp))

  val dlmP = for {
    v <- priorV
    seasonalP = DlmParameters(v = DenseMatrix(v),
                              w = DenseMatrix.eye[Double](7),
                              m0 = DenseVector.zeros[Double](7),
                              c0 = DenseMatrix.eye[Double](7))
  } yield List.fill(4)(seasonalP).reduce(Dlm.outerSumParameters)

  val initP: Rand[DlmFsvParameters] = for {
    fsv <- fsvP
    dlm <- dlmP
  } yield DlmFsvParameters(dlm, fsv)

  val id = "new_new_emote_2604"

  val d = trainingData.filter(_.sensorId == id).map(envToData)

  val iters = DlmFsvSystem.sample(priorBeta,
                                  priorSigmaEta,
                                  priorPhi,
                                  priorMu,
                                  priorSigma,
                                  priorV,
                                  d,
                                  dlmComp,
                                  initP.draw)

  Streaming
    .writeParallelChain(iters,
                        2,
                        100000,
                        s"examples/data/uo_gibbs_one_factor_${id}",
                        (s: DlmFsvSystem.State) => s.p.toList)
    .runWith(Sink.onComplete { s =>
      println(s)
      system.terminate()
    })
}

object InterpolateUo extends App with EmoteData {
  implicit val system = ActorSystem("interpolate-uo")
  implicit val materializer = ActorMaterializer()

  Source(testData)
    .filter(_.sensorId == "new_new_emote_2604")
    .map(envToData)
    .grouped(10000)
    .mapAsync(1) { d =>
      val file =
        "examples/data/uo_gibbs_two_factors_new_new_emote_2604_10000_0.csv"

      val ps: Future[DlmFsvParameters] = Streaming
        .readCsv(file)
        .map(_.map(_.toDouble).toList)
        .drop(5000)
        .map(x => DlmFsvSystem.paramsFromList(4, 28, 2)(x))
        .via(Streaming.meanDlmFsvSystemParameters(4, 28, 2))
        .runWith(Sink.head)

      val out = new java.io.File("examples/data/interpolate_urbanobservatory_2604_august.csv")
      val headers = rfc.withHeader(false)

      for {
        params <- ps
        times = d.map(_.time)
        iters: Iterator[Vector[DenseVector[Double]]] = DlmFsvSystem
          .sampleStateAr(d.toVector, dlmComp, params)
          .steps
          .map { (s: DlmFsvSystem.State) =>
            val vol = s.volatility.map(x => (x.time, x.sample))
            val st = s.theta.map(x => (x.time, x.sample))
            DlmFsv.obsVolatility(vol, st, dlmComp, params).
              map(_._2)
          }.
        take(1000)
        summary = iters.toVector.transpose.
        map(Dglm.meanCovSamples).
        map { case (mean, cov) => (mean.data ++ cov.data).toList }
        // io <- Source.fromIterator(() => iters).
        //   take(100).
        //   runWith(Streaming.writeChainSink(outfile, x => x))
        io <- Future.successful(out.writeCsv(times zip summary, headers))
      } yield io
    }
    .runWith(Sink.onComplete { s =>
      println(s)
      system.terminate()
    })
}

object UoResiduals extends App {
  implicit val system = ActorSystem("factor_sv")
  implicit val materializer = ActorMaterializer()

  val format = DateTimeFormatter.ISO_DATE_TIME
  implicit val codec: CellCodec[LocalDateTime] =
    localDateTimeCodec(format)

  val rawData = Paths.get("examples/data/uo_dlm_residuals.csv")
  val reader = rawData
    .asCsvReader[(String, LocalDateTime, List[Option[Double]])](rfc.withHeader)
  val data = reader.collect {
    case Right(a) =>
      (a._1,
       Data(a._2.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0),
            DenseVector(a._3.toArray)))
    case Left(ex) => throw new Exception(s"balls $ex")
  }.toVector

  val priorBeta = Gaussian(0.0, 1.0)
  val priorSigmaEta = InverseGamma(2, 2.0)
  val priorPhi = Gaussian(0.8, 0.2)
  val priorMu = Gaussian(2.0, 1.0)
  val priorSigma = InverseGamma(10, 2.0)

  val volP = for {
    se <- priorSigmaEta
    phi <- priorPhi
    mu <- priorMu
  } yield SvParameters(phi, mu, se)

  val fsvP = for {
    bij <- priorBeta
    sigma <- priorSigma
    vp <- volP
  } yield
    FsvParameters(DenseMatrix.eye[Double](4) * sigma,
                  FactorSv.buildBeta(4, 1, bij),
                  Vector.fill(1)(vp))

  def formatParameters(s: FactorSv.State) = s.params.toList

  data
    .groupBy(_._1)
    .map {
      case (name, ds) =>
        val iters = FactorSv
          .sampleAr(priorBeta,
                    priorSigmaEta,
                    priorMu,
                    priorPhi,
                    priorSigma,
                    ds.map(_._2),
                    fsvP.draw)

        Streaming
          .writeParallelChain(iters,
                              2,
                              100000,
                              s"examples/data/uo_residuals_factor_$name",
                              formatParameters)
          .runWith(Sink.head)
    }
    .toVector
    .sequence
    .onComplete(_ => system.terminate())
}

object UoSystem extends App {
  implicit val system = ActorSystem("factor_sv")
  implicit val materializer = ActorMaterializer()

  val format = DateTimeFormatter.ISO_DATE_TIME
  implicit val codec: CellCodec[LocalDateTime] =
    localDateTimeCodec(format)

  val rawData = Paths.get("examples/data/smooth_urbanobservatory.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) =>
      Data(a.head, DenseVector(a.tail.toArray.map(_.some)))
    case Left(ex) => throw new Exception(s"balls $ex")
  }.toVector

  val priorBeta = Gaussian(0.0, 1.0)
  val priorSigmaEta = InverseGamma(2, 2.0)
  val priorPhi = Gaussian(0.8, 0.1)
  val priorMu = Gaussian(2.0, 1.0)
  val priorSigma = InverseGamma(10, 2.0)

  val volP = for {
    se <- priorSigmaEta
    phi <- priorPhi
    mu <- priorMu
  } yield SvParameters(phi, mu, se)

  val fsvP = for {
    bij <- priorBeta
    sigma <- priorSigma
    vp <- volP
  } yield
    FsvParameters(DenseMatrix.eye[Double](28) * sigma,
                  FactorSv.buildBeta(28, 2, bij),
                  Vector.fill(2)(vp))

  def formatParameters(s: FactorSv.State) = s.params.toList

  val iters = FactorSv
    .sampleAr(priorBeta,
              priorSigmaEta,
              priorMu,
              priorPhi,
              priorSigma,
              data,
              fsvP.draw)

  Streaming
    .writeParallelChain(iters,
                        2,
                        100000,
                        s"examples/data/uo_residuals_system_factor_2604",
                        formatParameters)
    .runWith(Sink.onComplete(_ => system.terminate()))

}

object ForecastUo extends App {
  implicit val system = ActorSystem("forecast-uo")
  implicit val materializer = ActorMaterializer()

  val dlm = Dlm.polynomial(1) |+|
    Dlm.seasonal(24, 3) |+| Dlm.seasonal(24 * 7, 3)
  val mod = Dglm.studentT(3, dlm)

  val file = "examples/data/uo_cont_gibbs_two_factors_0.csv"

  // read parameters from file
  val ps: Future[DlmFsvParameters] = Streaming
    .readCsv(file)
    .map(_.map(_.toDouble).toList)
    .map(x => DlmFsvParameters.fromList(4, 16, 4, 2)(x))
    .via(Streaming.meanDlmFsvParameters(4, 16, 4, 2))
    .runWith(Sink.head)

  val out = new java.io.File("examples/data/forecast_uo.csv")
  val headers = rfc.withHeader("time", "mean", "lower", "upper")

  val n = 1000
}
