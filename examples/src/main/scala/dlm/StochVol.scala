package examples.dlm

import dlm.core.model._
import breeze.stats.distributions._
import breeze.linalg.DenseVector
import breeze.stats.mean
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
  val sims = StochasticVolatility.simulate(p).
    steps.take(10000).toVector

  // write to file
  val out = new java.io.File("examples/data/sv_sims.csv")
  val headers = rfc.withHeader("time", "observation", "log_volatility")
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
    case Right(a) => Data(a.head, DenseVector(a(1).some))
  }.toVector

  val priorSigma = InverseGamma(10, 2)
  val priorPhi = new Beta(5, 2)
  val priorMu = Gaussian(1.0, 1.0)

  val iters = StochasticVolatility.sampleAr(priorPhi, priorMu, priorSigma, p, data)

  def formatParameters(s: StochVolState) = {
    List(s.params.phi, s.params.mu, s.params.sigmaEta, s.accepted)
  }

  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/sv_params", formatParameters).
    runWith(Sink.onComplete(_ => system.terminate()))
}

object FitSvKnots extends App {
  implicit val system = ActorSystem("stochastic_volatility_knots")
  implicit val materializer = ActorMaterializer()

  val p = SvParameters(0.8, 1.0, 0.2)

  val rawData = Paths.get("examples/data/sv_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Data(a.head, DenseVector(a(1).some))
  }.toVector

  val priorSigma = InverseGamma(10, 2)
  val priorPhi = Gaussian(0.8, 0.2)
  val priorMu = Gaussian(1.0, 1.0)

  val iters = StochasticVolatilityKnots.sample(priorPhi, priorMu, priorSigma, data, p)

  def formatParameters(s: StochVolState) = {
    List(s.params.phi, s.params.mu, s.params.sigmaEta)
  }

  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/sv_knot_params", formatParameters).
    runWith(Sink.onComplete(_ => system.terminate()))
}

object FitSvKnotsBeta extends App {
  implicit val system = ActorSystem("stochastic_volatility_knots")
  implicit val materializer = ActorMaterializer()

  val p = SvParameters(0.8, 1.0, 0.2)

  val rawData = Paths.get("examples/data/sv_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Data(a.head, DenseVector(a(1).some))
  }.toVector

  val priorPrec = InverseGamma(10, 2)
  val priorPhi = new Beta(5.0, 2.0)
  val priorMu = Gaussian(1.0, 1.0)

  val iters = StochasticVolatilityKnots.sampleBeta(priorMu, priorPrec, priorPhi, data, p)

  def formatParameters(s: StochVolState) = {
    List(s.params.phi, s.params.mu, s.params.sigmaEta, s.accepted)
  }

  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/sv_knot_beta", formatParameters).
    runWith(Sink.onComplete(_ => system.terminate()))
} 

object SvSampleStateMixture extends App {
  val params = SvParameters(0.8, 1.0, 0.2)
  val rawData = Paths.get("examples/data/sv_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Data(a.head, DenseVector(a(1).some))
  }.toVector.
    take(1000)

  val initState = StochasticVolatilityKnots.initialState(params, data)

  // sample state mixture
  val p = StochasticVolatility.ar1DlmParams(params)
  val sampleState = StochasticVolatility.sampleState(data, p, params.phi,
        FilterAr.advanceState(params), FilterAr.backStep(params)) _

  val iters = MarkovChain(initState.draw)(sampleState).
    steps.
    take(1000).
    toVector

  def quantile[A: Ordering](xs: Seq[A], prob: Double): A = {
    val index = math.floor(xs.length * prob).toInt
    val ordered = xs.sorted
    ordered(index)
  }

  val summary = iters.transpose.map { x =>
    val sample = x.map(_.sample(0))
    (x.head.time, mean(sample), quantile(sample, 0.995), quantile(sample, 0.005))
  }

  // write state
  val out = new java.io.File("examples/data/sv_state_mixture.csv")
  val headers = rfc.withHeader("time", "mean", "upper", "lower")
  out.writeCsv(summary, headers)
}

object SvSampleStateKnots extends App {
  val params = SvParameters(0.8, 1.0, 0.2)
  val rawData = Paths.get("examples/data/sv_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Data(a.head, DenseVector(a(1).some))
  }.toVector.
    take(1000)

  val initState = StochasticVolatilityKnots.initialState(params, data).draw

  val sampleState = (st: Vector[SamplingState]) => for {
    knots <- StochasticVolatilityKnots.sampleKnots(10, 100)(data.size)
    state = StochasticVolatilityKnots.sampleState(data,
      FilterAr.advanceState, FilterAr.backStep, params, knots)(st)
  } yield state

  val iters = MarkovChain(initState)(sampleState).
    steps.
    take(1000).
    toVector

  def quantile[A: Ordering](xs: Seq[A], prob: Double): A = {
    val index = math.floor(xs.length * prob).toInt
    val ordered = xs.sorted
    ordered(index)
  }

  val summary = iters.transpose.map { x =>
    val sample = x.map(_.sample(0))
    (x.head.time, mean(sample), quantile(sample, 0.995), quantile(sample, 0.005))
  }

  // write state
  val out = new java.io.File("examples/data/sv_state_knots.csv")
  val headers = rfc.withHeader("time", "mean", "upper", "lower")
  out.writeCsv(summary, headers)
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
    case Right(a) => Data(a.head, DenseVector(a(1).some))
  }.toVector.
    drop(1)

  val priorSigma = InverseGamma(10.0, 1.0)
  val priorPhi = new Beta(2.0, 5.0)
  val priorMu = Gaussian(1.0, 1.0)

  val iters = StochasticVolatility.sampleOu(priorSigma, priorPhi, priorMu, p, data)

  def formatParameters(s: StochVolState) = {
    List(s.params.phi, s.params.mu, s.params.sigmaEta)
  }

  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/sv_ou_params", formatParameters).
    runWith(Sink.onComplete(_ => system.terminate()))
}

object RainierSv extends App {
  import com.stripe.rainier.compute._
  import com.stripe.rainier.core._
  import com.stripe.rainier.sampler._

  implicit val rng = ScalaRNG(4) // set a seed

  val rawData = Paths.get("examples/data/sv_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Data(a.head, DenseVector(a(1).some))
  }.toVector

  case class Parameters(phi: Real, mu: Real, sigma: Real)

  val prior = for {
    phi <- Beta(2, 5).param
    mu <- Normal(0, 1).param
    sigma <- LogNormal(0, 1).param
    a0 <- Normal(mu, sigma * sigma / (1 - phi * phi)).param
  } yield (Parameters(phi, mu, sigma), a0)

  def addTimePoint(
    params: RandomVariable[(Parameters, Real)],
    y: Double): RandomVariable[(Parameters, Real)] =
    for {
      (p, a) <- params
      a1 <- Normal(p.mu + p.phi * (a - p.mu), p.sigma).param
      sigma <- Normal(0, 1).param
      _ <- Normal(0, (a1 / 2).exp).fit(y)
    } yield (p, a1)

  val fullModel = data.map(_.observation(0).get).foldLeft(prior)(addTimePoint)

  val model = for {
    p <- fullModel
  } yield Map("mu" -> p._1.mu, "phi" -> p._1.phi, "sigma" -> p._1.sigma)

  val iters = model.sample(HMC(10), 1000, 100000, 10)

  val out = new java.io.File("examples/data/sv_rainier.csv")
  val headers = rfc.withHeader("mu", "phi", "sigma")

  out.writeCsv(iters.map(_.values), headers)
}
