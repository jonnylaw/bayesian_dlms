package examples.dlm

import dlm.core.model._
import breeze.stats.distributions._
import breeze.linalg.DenseVector
import breeze.stats.mean
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import kantan.csv.generic._
import akka.actor.ActorSystem
import akka.stream._
import scaladsl._

/**
  * Simulate a stochastic volatility model with an AR(1) latent state
  */
object SimulateSv extends App {
  val p = SvParameters(0.8, 2.0, 0.3)
  val sims = StochasticVolatility.simulate(p).
    steps.
    take(10000).
    toVector.
    tail

  // write to file
  val out = new java.io.File("examples/data/sv_sims.csv")
  val headers = rfc.withHeader("time", "observation", "log_volatility")
  def formatData(d: (Double, Option[Double], Double)) = d match {
    case (t, y, a) => List(t, y.get, a)
  }
  out.writeCsv(sims.map(formatData), headers)
}

object FitSv extends App {
  implicit val system = ActorSystem("stochastic_volatility")
  implicit val materializer = ActorMaterializer()

  case class SvSims(
    time: Double,
    observation: Double,
    volatility: Double
  )

  val rawData = Paths.get("examples/data/sv_sims.csv")
  val reader = rawData.asCsvReader[SvSims](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => (a.time, a.observation.some)
  }.toVector

  val priorPhi = Gaussian(0.8, 0.1)
  val priorMu = Gaussian(2.0, 1.0)
  val priorSigma = InverseGamma(2.0, 2.0)

  val iters = StochasticVolatility.
    sampleUni(priorPhi, priorMu, priorSigma, data)

  def formatParameters(s: StochVolState) = {
    List(s.params.phi, s.params.mu, s.params.sigmaEta)
  }

  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/sv_params", formatParameters).
    runWith(Sink.onComplete(_ => system.terminate()))
}

object FitSvKnots extends App {
  implicit val system = ActorSystem("stochastic_volatility_knots")
  implicit val materializer = ActorMaterializer()

  case class SvSims(
    time: Double,
    observation: Double,
    volatility: Double
  )
  val rawData = Paths.get("examples/data/sv_sims.csv")
  val reader = rawData.asCsvReader[SvSims](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => (a.time, a.observation.some)
  }.toVector

  val priorPhi = Gaussian(0.8, 0.1)
  val priorMu = Gaussian(2.0, 1.0)
  val priorSigma = InverseGamma(2, 2)

  val iters = StochasticVolatilityKnots.sampleParametersAr(priorPhi,
    priorMu, priorSigma, data)

  def formatParameters(s: StochVolState) = 
    s.params.toList


  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/sv_knot_params", formatParameters).
    runWith(Sink.onComplete(_ => system.terminate()))
}

object FitSvMixBeta extends App {
  implicit val system = ActorSystem("stochastic_volatility_knots")
  implicit val materializer = ActorMaterializer()

  val rawData = Paths.get("examples/data/sv_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => (a.head, a(1).some)
  }.toVector

  val priorPhi = new Beta(5.0, 2.0)
  val priorMu = Gaussian(2.0, 1.0)
  val priorSigma = InverseGamma(2, 2)

  val iters = StochasticVolatility.sampleBeta(priorPhi, priorMu,
                                              priorSigma, data)

  def formatParameters(s: StochVolState) = {
    List(s.params.phi, s.params.mu, s.params.sigmaEta)
  }

  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/sv_mix_beta", formatParameters).
    runWith(Sink.onComplete(_ => system.terminate()))
}

object SvSampleStateMixture extends App {
  implicit val system = ActorSystem("stochastic_volatility_knots")
  implicit val materializer = ActorMaterializer()

  val rawData = Paths.get("examples/data/sv_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => (a.head, a(1).some)
  }.toVector.
    take(1000)

  Streaming.
    readCsv("examples/data/sv_params_0.csv").
    drop(10000).
    via(Streaming.thinChain(20)).
    map(_.map(_.toDouble).toList).
    map(SvParameters.fromList).
    via(Streaming.meanSvParameters).
    map { params =>

      val initState = StochasticVolatilityKnots.initialStateAr(params, data)
      val iters = MarkovChain(initState.draw)(StochasticVolatility.sampleStateAr(data, params, _)).
        steps.
        take(1000).
        toVector

      val out = new java.io.File("examples/data/sv_state_mixture.csv")
      val headers = rfc.withHeader("time", "mean", "upper", "lower")

      val summary = iters.transpose.map { x =>
        val xt = x.map(_.sample)
        (x.head.time, mean(xt),
         Streaming.quantile(xt, 0.95), Streaming.quantile(xt, 0.05))
      }
      out.writeCsv(summary, headers)
    }.runWith(Sink.onComplete(_ => system.terminate()))
}

object SvSampleStateKnots extends App {
  implicit val system = ActorSystem("stochastic_volatility_knots")
  implicit val materializer = ActorMaterializer()

  import StochasticVolatilityKnots._

  val rawData = Paths.get("examples/data/sv_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => (a.head, a(1).some)
  }.toVector.
    take(1000)

  Streaming.
    readCsv("examples/data/sv_params_0.csv").
    drop(10000).
    via(Streaming.thinChain(20)).
    map(_.map(_.toDouble).toList).
    map(SvParameters.fromList).
    via(Streaming.meanSvParameters).
    map { p =>
      val sampleStep = (st: Array[FilterAr.SampleState]) => for {
        knots <- sampleKnots(10, 100, data.size)
        state = sampleState(ffbsAr, filterAr, sampleAr)(data, p, knots, st)
      } yield state

      val initState = initialStateAr(p, data).draw.toArray

      val iters = MarkovChain(initState)(sampleStep).
          steps.
          take(1000).
          map(_.map(x => (x.time, x.sample)).toVector).
          toVector

      val out = new java.io.File("examples/data/sv_state_knots.csv")
      val headers = rfc.withHeader("time", "mean", "upper", "lower")

         val summary = iters.transpose.map { x =>
           val xt = x.map(_._2)
           (x.head._1, mean(xt),
            Streaming.quantile(xt, 0.95), Streaming.quantile(xt, 0.05))
         }
         out.writeCsv(summary, headers)
    }.runWith(Sink.onComplete(_ => system.terminate()))
}

object SimulateOu extends App {
  // simulate data
  val p = SvParameters(0.2, 1.0, 0.3)
  val times = Stream.iterate(0.1)(t => t + scala.util.Random.nextDouble())
  val sims = StochasticVolatility.simOu(p, times).
    take(10000).toVector

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

  val priorPhi = new Beta(2.0, 5.0)
  val priorMu = Gaussian(1.0, 1.0)
  val priorSigma = InverseGamma(10.0, 1.0)

  val iters = StochasticVolatilityKnots.sampleOu(priorPhi, priorMu,
    priorSigma, data, p)

  def formatParameters(s: StochasticVolatilityKnots.OuSvState) = {
    List(s.params.phi, s.params.mu, s.params.sigmaEta) :::
    s.accepted.data.map(_.toDouble).toList
  }

  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/sv_ou_params", formatParameters).
    runWith(Sink.onComplete(_ => system.terminate()))
}
