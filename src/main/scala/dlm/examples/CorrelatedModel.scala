package dlm.examples

import dlm.model._
import cats.implicits._
import Dlm._
import breeze.linalg._
import breeze.stats.distributions._
import java.nio.file.Paths
import kantan.csv._
import kantan.csv.ops._

trait CorrelatedModel {
  // first define two models, one for each time series
  val mod1 = polynomial(1)
  val mod2 = polynomial(1)

  // combine the models in an outer product
  val model = Dlm.outerSumModel(mod1, mod2)

  // specify the parameters for the joint model
  val v = diag(DenseVector(1.0, 4.0))
  val w = DenseMatrix((0.75, 0.5), (0.5, 1.25))
  val c0 = DenseMatrix.eye[Double](2)

  val p = Parameters(v, w, DenseVector.zeros[Double](2), c0)
}

trait CorrelatedData {
  val rawData = Paths.get("data/correlated_dlm.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.
    collect { 
      case Success(a) => Data(a.head.toInt, DenseVector(a(1), a(2)).some)
    }.
    toArray
}

object SimulateCorrelated extends App with CorrelatedModel {
  val sims = Dlm.simulate(0, model, p).
    steps.
    take(1000)

  val out = new java.io.File("data/correlated_dlm.csv")
  val headers = rfc.withHeader("time", "observation_1", "observation_2", "state_1", "state_2")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatData(d: (Data, DenseVector[Double])) = d match {
    case (Data(t, y), x) =>
      List(t.toDouble) ++ y.map(_.data).get ++ x.data
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

object FilterCorrelatedDlm extends App with CorrelatedModel with CorrelatedData {
  val filtered = KalmanFilter.kalmanFilter(model, data, p)

  val out = new java.io.File("data/correlated_dlm_filtered.csv")

  def formatFiltered(f: KalmanFilter.State) = {
    (f.time, f.mt(0), f.ct.data(0), 
      f.y.map(_(0)), f.cov.map(_.data(0)))
  }

  val headers = rfc.withHeader("time", "state_mean", 
    "state_variance", "one_step_forecast", "one_step_variance")

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object GibbsCorrelated extends App with CorrelatedModel with CorrelatedData {

  val iters = GibbsWishart.sample(model, InverseGamma(1.0, 1.0), InverseWishart(3, DenseMatrix.eye[Double](2)), p, data).
    steps.
    take(10000)

  val out = new java.io.File("data/correlated_dlm_gibbs.csv")
  val headers = rfc.withHeader("V1", "V2", "V3", "V4", "W1", "W2", "W3", "W4")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatParameters(p: Parameters) = {
    (p.v.data ++ p.w.data).toList
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.p))
  }

  writer.close()
}

object FirstOrderLinearTrendDlm extends App {
  val mod1 = polynomial(1)
  val mod2 = polynomial(2)

  val composedModel = Dlm.outerSumModel(mod1, mod2)

  val p = Parameters(
    v = diag(DenseVector(1.0, 2.0)),
    w = diag(DenseVector(2.0, 3.0, 1.0)),
    m0 = DenseVector.zeros[Double](3),
    c0 = DenseMatrix.eye[Double](3)
  )

  val sims = Dlm.simulate(0, composedModel, p).
    steps.
    take(1000)

  val out = new java.io.File("data/first_order_and_linear_trend.csv")
  val headers = rfc.withHeader("time", "observation_1", "observation_2", "state_1", "state_2", "state_3")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatData(d: (Data, DenseVector[Double])) = d match {
    case (Data(t, y), x) =>
      List(t.toDouble) ++ y.map(_.data).get ++ x.data
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}
