package examples.dlm

import core.dlm.model._
import Dlm._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.{RandBasis}
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._

trait FirstOrderDlm {
  val mod = DlmModel(
    f = (t: Double) => DenseMatrix((1.0)),
    g = (t: Double) => DenseMatrix((1.0))
  )
  val p = DlmParameters(v = DenseMatrix(2.0),
                        w = DenseMatrix(3.0),
                        DenseVector(0.0),
                        DenseMatrix(1.0))
}

trait SimulatedData {
  val rawData = Paths.get("core/data/first_order_dlm.csv")
  val reader = rawData.asCsvReader[(Double, Double, Double)](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Data(a._1, DenseVector(a._2.some))
  }.toVector
}

object SimulateDlm extends App with FirstOrderDlm {
  val sims = simulateRegular(mod, p, 1.0).steps.take(300)

  val out = new java.io.File("core/data/first_order_dlm.csv")
  val headers = rfc.withHeader("time", "observation", "state")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatData(d: (Data, DenseVector[Double])) = d match {
    case (Data(t, y), x) =>
      t :: KalmanFilter.flattenObs(y).data.toList ::: x.data.toList
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

object FilterDlm extends App with FirstOrderDlm with SimulatedData {
  val filtered = SvdFilter.filter(mod, data, p, SvdFilter.advanceState(p, mod.g))

  val out = new java.io.File("core/data/first_order_dlm_filtered.csv")

  def formatFiltered(f: SvdState) = {
    val ct = f.uc * diag(f.dc) * f.uc.t
    List(f.time, f.mt(0), ct(0, 0), f.ft(0))
  }
  val headers = rfc.withHeader("time",
                               "state_mean",
                               "state_variance",
                               "one_step_forecast",
                               "one_step_variance")

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object SmoothDlm extends App with FirstOrderDlm with SimulatedData {
  val filtered = KalmanFilter.filter(mod, data, p, KalmanFilter.advanceState(p, mod.g))
  val smoothed = Smoothing.backwardsSmoother(mod)(filtered)

  val out = new java.io.File("core/data/first_order_dlm_smoothed.csv")

  def formatSmoothed(s: Smoothing.SmoothingState) =
    (s.time, s.mean(0), s.covariance.data(0))

  out.writeCsv(smoothed.map(formatSmoothed),
               rfc.withHeader("time", "smoothed_mean", "smoothed_variance"))
}

object SampleStates extends App with FirstOrderDlm with SimulatedData {
  implicit val basis = RandBasis.withSeed(7)

  val svdSampled = SvdSampler.ffbs(mod, data, p, SvdFilter.advanceState(p, mod.g)).sample(1000)
  val meanStateSvd = SvdSampler.meanState(svdSampled)
  val outSvd = new java.io.File("core/data/first_order_state_svd_sample.csv")

  outSvd.writeCsv(meanStateSvd, rfc.withHeader("time", "sampled_mean"))

  val sampled = Smoothing.ffbs(mod, data,
    KalmanFilter.advanceState(p, mod.g),
    Smoothing.step(mod, p.w), p).sample(1000)
  val meanState = SvdSampler.meanState(sampled)
  val out = new java.io.File("core/data/first_order_state_sample.csv")

  out.writeCsv(meanState, rfc.withHeader("time", "sampled_mean"))
}

object GibbsParameters extends App with FirstOrderDlm with SimulatedData {
  val priorV = InverseGamma(4.0, 6.0)
  val priorW = InverseGamma(6.0, 15.0)

  val iters = GibbsSampling
    .sample(mod, priorV, priorW, p, data)
    .steps
    .take(10000)
    .map(_.p)

  // write iters to file
  val headers = rfc.withHeader("V", "W")
  def formatParameters(p: DlmParameters) = {
    List(p.v.data(0), p.w.data(0))
  }

  Streaming.writeChain(formatParameters,
                       "core/data/first_order_dlm_gibbs.csv",
                       headers)(iters)
}
