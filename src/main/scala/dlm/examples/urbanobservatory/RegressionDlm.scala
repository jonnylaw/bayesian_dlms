package dlm.examples.urbanobservatory

import dlm.model._
import Dlm._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.{Gaussian, Rand}
import java.nio.file.Paths
import breeze.numerics.exp
import kantan.csv._
import kantan.csv.ops._

object RegressionDlm extends App with ObservedData {
  // interpolate missing Humidity values
  val humidity = {
    val y = data.map(_.observation.map(a => a(0)))
    TimeSeries.interpolate(y).map(DenseVector(_))
  }

  val model = Dlm.regression(humidity)

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

  val iters = GibbsSampling.gibbsSamples(model, InverseGamma(1.0, 1.0), InverseGamma(1.0, 1.0), p, temperature).
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
