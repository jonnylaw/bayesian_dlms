package dlm.examples.urbanobservatory

import dlm.model._
import Dlm._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.{Gamma, Gaussian, Rand}
import java.nio.file.Paths
import breeze.numerics.exp
import kantan.csv._
import kantan.csv.ops._

object RegressionDlm extends App with ObservedData {
  val humidity = for {
    d <- data
    y = d.observation
    h = y.map(a => DenseVector(a(0)))
  } yield Data(d.time, h)

  val model = Dlm.regression(TimeSeries.interpolate(humidity.map(_.observation)))

  val p = Parameters(
    DenseMatrix.eye[Double](1) * 50.0,
    diag(DenseVector(21.0, 7.0)),
    DenseVector.zeros[Double](2),
    DenseMatrix.eye[Double](2)
  )

  val temperature = for {
    d <- data
    y = d.observation
    t = y.map(a => DenseVector(a(1)))
  } yield Data(d.time, t)

  val iters = GibbsSampling.gibbsSamples(model, Gamma(1.0, 1.0), Gamma(1.0, 1.0), p, temperature).
    steps.
    take(1000000)

  val out = new java.io.File("data/humidity_temperature_regression_model_parameters.csv")
  val headers = rfc.withHeader("V", "W1", "W2", "W3", "W4")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatParameters(p: Parameters) = {
    (p.v(0, 0) +: p.w.data).toList
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.p))
  }

  writer.close()
}
