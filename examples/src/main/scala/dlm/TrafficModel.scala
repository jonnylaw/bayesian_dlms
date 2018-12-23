package examples.dlm

import dlm.core.model._
import breeze.linalg.{DenseMatrix, DenseVector, sum, diag}
import breeze.stats.distributions._
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import java.time._
import java.time.format._
import java.time.temporal.ChronoUnit
import kantan.csv.java8._
import akka.actor.ActorSystem
import akka.stream._
import scaladsl._
import scala.concurrent.Future

trait ReadTrafficData {
  val format = DateTimeFormatter.ISO_DATE_TIME
  implicit val codec: CellCodec[LocalDateTime] = localDateTimeCodec(format)

  val rawData = Paths.get("examples/data/training_traffic.csv")
  val reader = rawData.asCsvReader[(LocalDateTime, Double)](rfc.withHeader)
  val data = reader
    .collect {
      case Right(a) => a
    }
    .toVector
    .map {
      case (t, x) =>
        Data(t.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0),
             DenseVector(x.some))
    }

  val rawData1 = Paths.get("examples/data/test_traffic.csv")
  val reader1 = rawData1.asCsvReader[(LocalDateTime, Double)](rfc.withHeader)
  val test = reader1
    .collect { case Right(a) => a }
    .toVector
    .map { x =>
      Data(x._1.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0),
           DenseVector(x._2.some))
    }
}

object TrafficPoisson extends App with ReadTrafficData {
  implicit val system = ActorSystem("traffic-poisson")
  implicit val materializer = ActorMaterializer()

  // read particles and proposal matrix from command line
  // val (n, delta) = (args.lift(0).map(_.toInt).getOrElse(500),
  //   args(1).lift(1).map(_.toDouble).getOrElse(0.05))

  val (n, delta) = (500, 0.01)

  val mod = Dglm.poisson(Dlm.polynomial(1) |+| Dlm.seasonal(24, 4))
  val params = DlmParameters(DenseMatrix(2.0),
                             diag(DenseVector.fill(9)(0.05)),
                             DenseVector.fill(9)(0.0),
                             diag(DenseVector.fill(9)(10.0)))

  def prior(p: DlmParameters) =
    diag(p.w).map(wi => InverseGamma(11.0, 1.0).logPdf(wi)).sum

  def proposal(delta: Double)(p: DlmParameters): Rand[DlmParameters] =
    for {
      propW <- Metropolis.proposeDiagonalMatrix(delta)(p.w)
    } yield p.copy(w = propW)

  val initState = Metropolis.State[DlmParameters](params, -1e99, 0)
  val ll = (p: DlmParameters) => AuxFilter.likelihood(mod, data, n)(p)

  val iters = MarkovChain(initState)(
    Metropolis.mStep[DlmParameters](proposal(delta), prior, ll))

  def diagonal(m: DenseMatrix[Double]) = {
    for {
      i <- (0 until m.cols)
    } yield m(i, i)
  }

  def format(s: Metropolis.State[DlmParameters]) = {
    diagonal(s.parameters.w).toList ++
      List(s.accepted.toDouble)
  }

  Streaming
    .writeParallelChain(
      iters,
      2,
      10000,
      s"examples/data/poisson_traffic_auxiliary_${n}_${delta}_pmmh",
      format)
    .runWith(Sink.onComplete(_ => system.terminate()))
}

object TrafficNegBin extends App with ReadTrafficData {
  implicit val system = ActorSystem("traffic-negbin")
  implicit val materializer = ActorMaterializer()

  // read particles and proposal matrix from command line
  // val (n, delta) = (args.lift(0).map(_.toInt).getOrElse(500),
  //   args(1).lift(1).map(_.toDouble).getOrElse(0.05))

  val (n, delta) = (500, 0.01)

  val mod = Dglm.negativeBinomial(Dlm.polynomial(1) |+| Dlm.seasonal(24, 4))
  val params = DlmParameters(DenseMatrix(2.0),
                             diag(DenseVector.fill(9)(0.05)),
                             DenseVector.fill(9)(0.0),
                             diag(DenseVector.fill(9)(10.0)))

  // very vague prior
  def prior(params: DlmParameters) = {
    val ws = diag(params.w).map(wi => InverseGamma(11.0, 1.0).logPdf(wi)).sum

    InverseGamma(2.0, 2.0).logPdf(params.v(0, 0)) + ws
  }

  def proposal(delta: Double)(p: DlmParameters): Rand[DlmParameters] =
    for {
      logv <- Metropolis.proposeDouble(delta)(p.v(0, 0))
      propW <- Metropolis.proposeDiagonalMatrix(delta)(p.w)
    } yield p.copy(v = DenseMatrix(logv), w = propW)

  val initState = Metropolis.State[DlmParameters](params, -1e99, 0)
  val ll = (p: DlmParameters) => AuxFilter.likelihood(mod, data, n)(p)

  val iters = MarkovChain(initState)(
    Metropolis.mStep[DlmParameters](proposal(delta), prior, ll))

  def format(s: Metropolis.State[DlmParameters]) = {
    DenseVector
      .vertcat(diag(s.parameters.v), diag(s.parameters.w))
      .data
      .toList ++
      List(s.accepted.toDouble)
  }

  Streaming
    .writeParallelChain(
      iters,
      2,
      10000,
      s"examples/data/negbin_traffic_auxiliary_${n}_${delta}_pmmh",
      format)
    .runWith(Sink.onComplete(_ => system.terminate()))
}

object OneStepForecastTraffic extends App with ReadTrafficData {
  implicit val system = ActorSystem("forecast-traffic")
  implicit val materializer = ActorMaterializer()

  val model = Dglm.negativeBinomial(Dlm.polynomial(1) |+| Dlm.seasonal(24, 4))

  // convert hours from the epoch to datetime
  def hoursToDatetime(hours: Double) = {
    val inst = Instant.ofEpochSecond(hours.toLong * 60 * 60)
    val tz = ZoneOffset.UTC
    LocalDateTime.ofInstant(inst, tz)
  }

  Streaming
    .readCsv("examples/data/negbin_traffic_auxiliary_500_0.05_pmmh_0.csv")
    .drop(1000)
    .map(_.map(_.toDouble).toList)
    .map(
      l =>
        DlmParameters(v = DenseMatrix(l.head),
                      w = diag(DenseVector(l.slice(1, 10).toArray)),
                      m0 = DenseVector.zeros[Double](9),
                      c0 = DenseMatrix.eye[Double](9) * 100.0))
    .via(Streaming.meanParameters(1, 9))
    .mapAsync(1) { p =>
      val out = new java.io.File("examples/data/forecast_traffic_negbin.csv")
      val headers = rfc.withHeader("time", "median", "lower", "upper")

      val n = 1000
      val pf = ParticleFilter(n, n, ParticleFilter.multinomialResample)
      val initState = pf.initialiseState(model, p, data)
      val lastState = data.foldLeft(initState)(pf.step(model, p)).state

      // println(s"Data has ${data.size} observations")
      // println(s"Test has ${test.size} observations")
      // lastState.take(10).foreach(println)

      val fcst = Dglm
        .forecastParticles(model, lastState, p, test.take(200))
        .map {
          case (t, x, f) =>
            (hoursToDatetime(t), Dglm.medianAndIntervals(0.75)(f))
        }
        .map { case (t, (f, l, u)) => (t, f(0), l(0), u(0)) }

      Future.successful(out.writeCsv(fcst, headers))
    }
    .runWith(Sink.onComplete { s =>
      println(s)
      system.terminate()
    })
}
