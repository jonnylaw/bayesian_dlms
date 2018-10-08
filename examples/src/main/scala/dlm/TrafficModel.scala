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

trait ReadTrafficData {
  val format = DateTimeFormatter.ISO_DATE_TIME
  implicit val codec: CellCodec[LocalDateTime] = localDateTimeCodec(format)

  val rawData = Paths.get("examples/data/training_traffic.csv")
  val reader = rawData.asCsvReader[(LocalDateTime, Double)](rfc.withHeader)
  val data = reader
    .collect {
      case Right(a) => a._2
    }
    .toVector
    .zipWithIndex
    .map { case (x, t) => Data(t, DenseVector(x.some)) }
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
    diag(p.w).
      map(wi => InverseGamma(11.0, 1.0).logPdf(wi)).
      sum

  def proposal(delta: Double)(p: DlmParameters): Rand[DlmParameters] = for {
      propW <- Metropolis.proposeDiagonalMatrix(delta)(p.w)
    } yield p.copy(w = propW)

  val initState = Metropolis.State[DlmParameters](params, -1e99, 0)
  val ll = (p: DlmParameters) =>
    AuxFilter.likelihood(mod, data, n)(p)

  val iters = MarkovChain(initState)(Metropolis.mStep[DlmParameters](proposal(delta), prior, ll))

  def diagonal(m: DenseMatrix[Double]) = {
    for {
      i <- (0 until m.cols)
    } yield m(i,i)
  }

  def format(s: Metropolis.State[DlmParameters]) = {
    diagonal(s.parameters.w).toList ++
    List(s.accepted.toDouble)
  }

  Streaming
    .writeParallelChain(iters, 2, 10000, s"examples/data/poisson_traffic_auxiliary_${n}_${delta}_pmmh", format)
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
  val params = DlmParameters(
    DenseMatrix(2.0),
    diag(DenseVector.fill(9)(0.05)),
    DenseVector.fill(9)(0.0),
    diag(DenseVector.fill(9)(10.0)))

  // very vague prior
  def prior(params: DlmParameters) = {
    val ws = diag(params.w).
      map(wi => InverseGamma(11.0, 1.0).logPdf(wi)).
      sum

    InverseGamma(2.0, 2.0).logPdf(params.v(0,0)) + ws
  }

  def proposal(delta: Double)(p: DlmParameters): Rand[DlmParameters] = for {
      logv <- Metropolis.proposeDouble(delta)(p.v(0,0))
      propW <- Metropolis.proposeDiagonalMatrix(delta)(p.w)
    } yield p.copy(v = DenseMatrix(logv), w = propW)

  val initState = Metropolis.State[DlmParameters](params, -1e99, 0)
  val ll = (p: DlmParameters) =>
    AuxFilter.likelihood(mod, data, n)(p)

  val iters = MarkovChain(initState)(Metropolis.mStep[DlmParameters](proposal(delta), prior, ll))

  def format(s: Metropolis.State[DlmParameters]) = {
    DenseVector.vertcat(diag(s.parameters.v),
      diag(s.parameters.w)).data.toList ++
    List(s.accepted.toDouble)
  }

  Streaming
    .writeParallelChain(iters, 2, 10000, s"examples/data/negbin_traffic_auxiliary_${n}_${delta}_pmmh", format)
    .runWith(Sink.onComplete(_ => system.terminate()))
}

object OneStepForecastTraffic extends App with ReadTrafficData {
  val model = Dglm.negativeBinomial(Dlm.polynomial(1) |+| Dlm.seasonal(24, 4))

  val params = DlmParameters(
    DenseMatrix(1.0),
    diag(DenseVector(0.5, 0.6, 0.5, 0.4, 0.4,
      0.5, 0.5, 0.25, 0.25)),
    DenseVector(2.3, -3.0, 3.3, 0.5, 3.1, 2.3, 0.8, 0.2, -0.9),
    diag(DenseVector.fill(9)(1.0)))

  val pf = ParticleFilter(500, ParticleFilter.multinomialResample)
  val init = pf.initialiseState(model, params, data)
  val filtered = data.take(2000).foldLeft(init)(pf.step(model, params))

  val forecast = Dglm.forecastParticles(model, filtered.state,
    params, data.drop(2000)).
    map { case (t, x, f) => (t, Dglm.meanAndIntervals(f)) }.
    map { case (t, (f, l, u)) => (t, f(0), l(0), u(0)) }

  val out = new java.io.File("examples/data/forecast_traffic_negbin.csv")
  val headers = rfc.withHeader("time", "mean", "lower", "upper")
  out.writeCsv(forecast, headers)
}
