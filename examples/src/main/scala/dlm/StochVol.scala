package examples.dlm

import core.dlm.model._
import breeze.stats.distributions._
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import plot.{Diagnostics, TimeSeries}
import com.cibo.evilplot.plot.aesthetics.DefaultTheme._

/**
  * Simulate a stochastic volatility model with an AR(1) latent state
  */
object SimulateSv extends App {
  // simulate data
  val p = SvParameters(0.8, 1.0, 0.2)
  val sims = StochasticVolatility.simulate(p).steps.take(300).toVector

  // write to file
  val out = new java.io.File("core/data/sv_sims.csv")
  val headers = rfc.withHeader("time", "observation", "log-variance")
  def formatData(d: (Double, Option[Double], Double)) = d match {
    case (t, y, a) => List(t, y.getOrElse(0.0), a)
  }
  out.writeCsv(sims.map(formatData), headers)
}

object FitSv extends App {
  val p = SvParameters(0.8, 1.0, 0.2)

  val rawData = Paths.get("core/data/sv_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => (a.head, a(1).some)
  }.toVector

  val priorSigma = InverseGamma(10, 1)
  val priorPhi = new Beta(5, 2)
  val priorMu = Gaussian(0.0, 3.0)

  val iters = StochasticVolatility
    .sampleAr(priorSigma, priorPhi, priorMu, p, data)
    .steps
    .drop(10000)
    .take(90000)
    .map(_.params)

  // write iters to file
  val headers = rfc.withHeader("phi", "mu", "sigma")
  val out = new java.io.File("core/data/sv_params.csv")
  val writer = out.asCsvWriter[Vector[Double]](headers)

  def formatParameters(p: SvParameters): Vector[Double] = {
    Vector(p.phi, p.mu, p.sigmaEta)
  }

  while (iters.hasNext) {
    writer.write(formatParameters(iters.next))
  }

  writer.close()
}

