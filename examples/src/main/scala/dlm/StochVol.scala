package examples.dlm

import core.dlm.model._
import breeze.stats.distributions._
import breeze.linalg.DenseVector
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import akka.actor.ActorSystem
import akka.stream._
import scaladsl._

/**
  * Simulate a stochastic volatility model with an AR(1) latent state
  */
object SimulateSv extends App {
  // simulate data
  val p = SvParameters(0.8, 1.0, 0.2)
  val sims = StochasticVolatility.simulate(p).steps.take(3000).toVector

  // write to file
  val out = new java.io.File("examples/data/sv_sims.csv")
  val headers = rfc.withHeader("time", "observation", "log-variance")
  def formatData(d: (Double, Option[Double], Double)) = d match {
    case (t, y, a) => List(t, y.getOrElse(0.0), a)
  }
  out.writeCsv(sims.map(formatData), headers)
}

object FitSv extends App {
  implicit val system = ActorSystem("stochastic_volatility")
  implicit val materializer = ActorMaterializer()

  val p = SvParameters(0.8, 1.0, 0.2)

  val rawData = Paths.get("examples/data/sv_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => (a.head, a(1).some)
  }.toVector

  val priorSigma = InverseGamma(10, 1)
  val priorPhi = new Beta(5, 2)
  val priorMu = Gaussian(0.0, 3.0)

  val iters = StochasticVolatility.sampleAr(priorSigma, priorPhi, priorMu, p, data)

  def formatParameters(s: StochasticVolatility.State) = {
    List(s.params.phi, s.params.mu, s.params.sigmaEta)
  }

  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/sv_params", formatParameters).
    runWith(Sink.onComplete(_ => system.terminate()))
}

object FitSvKnots extends App {
  implicit val system = ActorSystem("stochastic_volatility")
  implicit val materializer = ActorMaterializer()

  val p = SvParameters(0.8, 1.0, 0.2)

  val rawData = Paths.get("examples/data/sv_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Dlm.Data(a.head, DenseVector(a(1).some))
  }.toVector

  val priorSigma = InverseGamma(0.001, 0.001)
  val priorPhi = Gaussian(0.0, 10.0)
  val priorMu = Gaussian(0.0, 10.0)

  val iters = StochasticVolatilityKnots.sample(priorMu, priorSigma, priorPhi, data, p)

  def formatParameters(s: StochasticVolatilityKnots.State) = {
    List(s.params.phi, s.params.mu, s.params.sigmaEta)
  }

  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/sv_knot_params", formatParameters).
    runWith(Sink.onComplete(_ => system.terminate()))
}

object SimulateOu extends App {
  // simulate data
  val p = SvParameters(0.2, 1.0, 0.3)
  val times = Stream.iterate(0.1)(t => t + scala.util.Random.nextDouble())
  val sims = StochasticVolatility.simOu(p, times).take(3000).toVector

  // write to file
  val out = new java.io.File("examples/data/sv_ou_sims.csv")
  val headers = rfc.withHeader("time", "observation", "log-variance")
  def formatData(d: (Double, Option[Double], Double)) = d match {
    case (t, y, a) => List(t, y.getOrElse(0.0), a)
  }
  out.writeCsv(sims.map(formatData), headers)
}

object FitSvOu extends App {
  implicit val system = ActorSystem("sv-ou")
  implicit val materializer = ActorMaterializer()

  val p = SvParameters(0.2, 1.0, 0.3)

  val rawData = Paths.get("examples/data/sv_ou_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => (a.head, a(1).some)
  }.toVector.
    drop(1)

  val priorSigma = InverseGamma(10.0, 1.0)
  val priorPhi = new Beta(2.0, 5.0)
  val priorMu = Gaussian(1.0, 1.0)

  val iters = StochasticVolatility.sampleOu(priorSigma, priorPhi, priorMu, p, data)

  def formatParameters(s: StochasticVolatility.State) = {
    List(s.params.phi, s.params.mu, s.params.sigmaEta)
  }

  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/sv_ou_params", formatParameters).
    runWith(Sink.onComplete(_ => system.terminate()))
}

