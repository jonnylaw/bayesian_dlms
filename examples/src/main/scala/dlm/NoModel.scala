package examples.dlm

import dlm.core.model._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions._
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import java.time._
import java.time.format._
import kantan.csv.java8._
import scala.concurrent.ExecutionContext.Implicits.global
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
        Data(secondsToHours(a._1.toEpochSecond(ZoneOffset.UTC)),
          DenseVector(Some(a._2)))
    }
    .toStream
    .zipWithIndex
    .filter { case (_, i) => i % 10 == 0 }
    .map(_._1)
    .toVector
    .take(5000)

  val testData = Paths.get("examples/data/test_no.csv")
  val testReader = testData.asCsvReader[(LocalDateTime, Double)](rfc.withHeader)
  val test = testReader
    .collect {
      case Right(a) =>
        Data(secondsToHours(a._1.toEpochSecond(ZoneOffset.UTC)),
          DenseVector(Some(a._2)))
    }.toVector
}

object NoStudentTModel extends App with NoData {
  implicit val system = ActorSystem("no-student-t-gibbs")
  implicit val materializer = ActorMaterializer()

  val dlm = Dlm.polynomial(1) |+|
    Dlm.seasonal(24, 3) |+| Dlm.seasonal(24 * 7, 3)
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

  def formatParameters(s: StudentT.State) =
    s.nu.toDouble :: s.p.toList

  Streaming
    .writeParallelChain(iters, 2, 10000,
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

  def formatParameters(s: GibbsSampling.State) = 
    s.p.toList

  Streaming
    .writeParallelChain(iters, 2, 10000,
      "examples/data/no_dlm_seasonal_weekly", formatParameters)
    .runWith(Sink.onComplete(_ => system.terminate()))
}

object ForecastNoGaussian extends App with NoData {
  implicit val system = ActorSystem("forecast-no")
  implicit val materializer = ActorMaterializer()

  val dlm = Dlm.polynomial(1) |+|
    Dlm.seasonal(24, 3) |+| Dlm.seasonal(24 * 7, 3)

  Streaming.
    readCsv("examples/data/no_dlm_seasonal_weekly_0.csv").
    drop(1000).
    map(_.map(_.toDouble).toList).
    map(DlmParameters.fromList(1, 13)).
    via(Streaming.meanParameters(1, 13)).
    map { params =>

      val out = new java.io.File("examples/data/forecast_no_dlm.csv")
      val headers = rfc.withHeader("mean", "lower", "upper")

      val p = params.copy(
        m0 = DenseVector.zeros[Double](13),
        c0 = DenseMatrix.eye[Double](13) * 100.0)

      val kf = KalmanFilter(KalmanFilter.advanceState(p, dlm.g))
      val initState = kf.initialiseState(dlm, p, data)
      val lastState = data.foldLeft(initState)(kf.step(dlm, p))

      val newP = p.copy(m0 = lastState.mt, c0 = lastState.ct)

      val filtered = kf.filter(dlm, test, p)

      val summary = filtered.flatMap(x => (for {
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

object ForecastNo extends App with NoData {
  implicit val system = ActorSystem("forecast-no")
  implicit val materializer = ActorMaterializer()

  val dlm = Dlm.polynomial(1) |+|
    Dlm.seasonal(24, 3) |+| Dlm.seasonal(24 * 7, 3)
  val mod = Dglm.studentT(5, dlm)

  Streaming.
    readCsv("examples/data/no_dglm_seasonal_weekly_student_exact_0.csv").
    drop(1000).
    map(_.map(_.toDouble).toList).
    map(l => l.tail).
    map(DlmParameters.fromList(1, 13)).
    via(Streaming.meanParameters(1, 13)).
    map { params =>

      val out = new java.io.File("examples/data/forecast_no.csv")
      val headers = rfc.withHeader("time", "mean", "lower", "upper")

      val p = params.copy(
        m0 = DenseVector.zeros[Double](13),
        c0 = DenseMatrix.eye[Double](13) * 100.0)
      val n = 1000
      val pf = ParticleFilter(n, ParticleFilter.multinomialResample)
      val initState = pf.initialiseState(mod, p, data)
      val lastState = data.foldLeft(initState)(pf.step(mod, p)).state
      val fcst = Dglm.forecastParticles(mod, lastState, p, test).
        map { case (t, x, f) => (t, Dglm.meanAndIntervals(0.975)(f)) }.
        map { case (t, (f, l, u)) => (t, f(0), l(0), u(0)) }

      out.writeCsv(fcst, headers)
    }.
    runWith(Sink.onComplete{ s =>
      println(s)
      system.terminate()
    })
}
