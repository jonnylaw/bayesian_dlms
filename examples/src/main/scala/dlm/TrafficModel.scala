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

  val mod = Dglm.poisson(Dlm.polynomial(1) |+| Dlm.seasonal(24, 4))
  val params = DlmParameters(DenseMatrix(2.0),
    diag(DenseVector.fill(9)(0.05)),
    DenseVector.fill(9)(0.0),
    diag(DenseVector.fill(9)(10.0)))

  def prior(p: DlmParameters) =
    diag(p.w).
      map(wi => InverseGamma(0.001, 0.001).logPdf(wi)).
      sum

  def proposal(delta: Double)(p: DlmParameters): Rand[DlmParameters] = for {
      propW <- Metropolis.proposeDiagonalMatrix(delta)(p.w)
    } yield p.copy(w = propW)


  val iters = Metropolis.dglm(mod, data, proposal(0.01), prior, params, 300)

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
    .writeParallelChain(iters, 2, 100000, "examples/data/poisson_traffic_200_0.01_pmmh", format)
    .runWith(Sink.onComplete(_ => system.terminate()))
}

object TrafficNegBin extends App with ReadTrafficData {
  implicit val system = ActorSystem("traffic-negbin")
  implicit val materializer = ActorMaterializer()

  val mod = Dglm.negativeBinomial(Dlm.polynomial(1) |+| Dlm.seasonal(24, 4))
  val params = DlmParameters(
    DenseMatrix(2.0),
    diag(DenseVector.fill(9)(0.05)),
    DenseVector.fill(9)(0.0),
    diag(DenseVector.fill(9)(10.0)))

  def prior(params: DlmParameters) = {
    val ws = diag(params.w).
      map(wi => InverseGamma(0.001, 0.001).logPdf(wi)).
      sum

    InverseGamma(0.001, 0.001).logPdf(params.v(0,0)) + ws
  }

  def proposal(delta: Double)(p: DlmParameters): Rand[DlmParameters] = for {
      logv <- Metropolis.proposeDouble(delta)(p.v(0,0))
      propW <- Metropolis.proposeDiagonalMatrix(delta)(p.w)
    } yield p.copy(v = DenseMatrix(logv), w = propW)

  val iters = Metropolis.dglm(mod, data, proposal(0.01),
    prior, params, 300)

  def format(s: Metropolis.State[DlmParameters]) = {
    DenseVector.vertcat(diag(s.parameters.v),
      diag(s.parameters.w)).data.toList ++
    List(s.accepted.toDouble)
  }

  Streaming
    .writeParallelChain(iters, 2, 100000, "examples/data/negbin_traffic_300_0.01_pmmh", format)
    .runWith(Sink.onComplete(_ => system.terminate()))
}
