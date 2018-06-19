package core.dlm.examples

import core.dlm.model._
import breeze.stats.distributions._
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._

object SimulateSv extends App {
  val p = StochasticVolatility.Parameters(0.8, 0.0, 0.2)

  val sims = StochasticVolatility.simulate(p).
    steps.
    take(300)

  val out = new java.io.File("core/data/sv_sims.csv")
  val headers = rfc.withHeader("time", "observation", "log-variance")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatData(d: (Double, Option[Double], Double)) = d match {
    case (t, y, a) => List(t, y.getOrElse(0.0), a)
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

object FitSv extends App {
  val p = StochasticVolatility.Parameters(0.8, 0.0, 0.2)

  val rawData = Paths.get("core/data/sv_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.
    collect { 
      case Right(a) => (a.head, a(1).some)
    }.
    toVector

  val priorSigma = InverseGamma(10, 1)
  val priorPhi = new Beta(5, 2)

  val iters = StochasticVolatility.sample(priorSigma, priorPhi, p, data).
    steps.take(100000).map(_.params)

  // write iters to file
  val headers = rfc.withHeader("phi", "mu", "sigma")
  val out = new java.io.File("core/data/sv_params.csv")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatParameters(p: StochasticVolatility.Parameters): List[Double] = {
    p.phi :: p.mu :: p.sigmaEta :: Nil
  }

  while (iters.hasNext) {
    writer.write(formatParameters(iters.next))
  }

  writer.close()
}
