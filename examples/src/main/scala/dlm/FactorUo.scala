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
import scala.concurrent.Future

trait EmoteData {
  val format = DateTimeFormatter.ISO_DATE_TIME
  implicit val codec: CellCodec[LocalDateTime] =
    localDateTimeCodec(format)

  val tempDlm = Dlm.polynomial(1) |+| Dlm.seasonal(24, 3)
  val dlmComp = List.fill(4)(tempDlm).reduce(_ |*| _)

  case class EnvSensor(
    sensorId:   String,
    datetime:    LocalDateTime,
    co:          Option[Double],
    humidity:    Option[Double],
    no:          Option[Double],
    temperature: Option[Double])

  val rawData = Paths.get("examples/data/encoded_sensor_data.csv")
  val reader = rawData.asCsvReader[EnvSensor](rfc.withHeader)
  val data: Vector[EnvSensor] = reader.
    collect {
      case Right(a) => a
    }.
    toVector

  def envToData(a: EnvSensor): Data =
    Data(a.datetime.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0),
         DenseVector(a.co, a.humidity, a.no map math.log, a.temperature))
}

object FitFactorUo extends App with EmoteData {
  implicit val system = ActorSystem("dlm_fsv_uo")
  implicit val materializer = ActorMaterializer()

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

  val fsvP = for {
    bij <- priorBeta
    sigma <- priorSigma
    vp <- volP
  } yield FsvParameters(DenseMatrix.eye[Double](28) * sigma,
    FactorSv.buildBeta(28, 2, bij),
    Vector.fill(2)(vp))

  val dlmP = for {
    w <- priorW
    seasonalP = DlmParameters(
      v = DenseMatrix.eye[Double](1),
      w = diag(DenseVector.fill(7)(w)),
      m0 = DenseVector.zeros[Double](7),
      c0 = DenseMatrix.eye[Double](7))
  } yield List.fill(4)(seasonalP).
    reduce(Dlm.outerSumParameters)

  val initP: Rand[DlmFsvParameters] = for {
    fsv <- fsvP
    dlm <- dlmP
  } yield DlmFsvParameters(dlm, fsv)

  val id = "new_new_emote_93"

  val d = data.
    filter(_.sensorId == id).
    map(envToData).
    sortBy(_.time)

  val iters = DlmFsvSystem.sample(priorBeta, priorSigmaEta, priorPhi, priorMu,
                                  priorSigma, priorW, d.toVector, dlmComp, initP.draw)

  Streaming
    .writeParallelChain(iters, 2, 10000, s"examples/data/uo_gibbs_two_factors_${id}",
                        (s: DlmFsvSystem.State) => s.p.toList).
    runWith(Sink.onComplete { s =>
              println(s)
              system.terminate()
            })
}

object InterpolateUo extends App with EmoteData {
  implicit val system = ActorSystem("interpolate-uo")
  implicit val materializer = ActorMaterializer()

  Source(data).
    groupBy(10000, _.sensorId).
    fold(("Hello", Vector.empty[Data]))((l, r) =>
      (r.sensorId, l._2 :+ envToData(r))).
    mergeSubstreams.
    mapAsync(1) { case (id, d) =>

      val file = s"examples/data/uo_gibbs_two_factors_${id}_0.csv"

      val ps: Future[DlmFsvParameters] = Streaming.readCsv(file).
        map(_.map(_.toDouble).toList).
        drop(1000).
        map(x => DlmFsvSystem.paramsFromList(4, 28, 2)(x)).
        via(Streaming.meanDlmFsvSystemParameters(4, 28, 2)).
        runWith(Sink.head)

      val out = new java.io.File(s"examples/data/interpolate_urbanobservatory_${id}.csv")
      val headers = rfc.withHeader(false)

      // use mcmc to sample the missing observations
      // val iters = DlmFsvSystem.sampleState(testData, dlmComp, params).
      //   steps.
      //   take(1000).
      //   map { (s: DlmFsv.State) =>
      //     val vol = s.volatility.map(x => (x.time, x.sample))
      //     val st = s.theta.map(x => (x.time, x.sample))
      //     DlmFsv.obsVolatility(vol, st, dlmComp, params)
      //   }.
      //   map(s => s.map { case (t, y) => (t, Some(y(0))) }).
      //   toVector

      // // calculate credible intervals
      // val summary = DlmFsvSystem.summariseInterpolation(iters, 0.995)

      // write interpolated data
      Future.successful("hello")

    }.runWith(Sink.onComplete(_ => system.terminate()))
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
