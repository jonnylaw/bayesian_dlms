package dlm.examples

import dlm.model._
import Dlm._
import MetropolisHastings._
import GibbsSampling._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Gamma
import java.nio.file.Paths
import java.io.{File, PrintWriter}
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import kantan.csv.generic._

trait FirstOrderDlm {
  val mod = Model(
    f = (t: Time) => DenseMatrix((1.0)), 
    g = (t: Time) => DenseMatrix((1.0))
  )
  val p = Parameters(
    DenseMatrix(3.0), 
    DenseMatrix(1.0), 
    DenseVector(0.0), 
    DenseMatrix(1.0))
}

trait SimulatedData {
  val rawData = Paths.get("data/FirstOrderDlm.csv")
  val reader = rawData.asCsvReader[(Time, Double, Double)](rfc.withHeader)
  val data = reader.
    collect { 
      case Success(a) => Data(a._1, Some(a._2).map(DenseVector(_)))
    }.
    toArray
}

object SimulateDlm extends App with FirstOrderDlm {
  val sims = simulate(0, mod, p).
    steps.
    take(1000)

  val out = new java.io.File("data/FirstOrderDlm.csv")
  val writer = out.asCsvWriter[(Time, Option[Double], Double)](rfc.withHeader("time", "observation", "state"))

  def formatData(d: (Data, DenseVector[Double])) = d match {
    case (Data(t, y), x) =>
      (t, y.map(x => x(0)), x(0))
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

object FilterDlm extends App with FirstOrderDlm with SimulatedData {
  val filtered = KalmanFilter.kalmanFilter(mod, data, p)

  val out = new java.io.File("data/FirstOrderDlmFiltered.csv")

  def formatFiltered(f: KalmanFilter.KfState) = {
    (f.time, f.statePosterior.mean(0), f.statePosterior.covariance.data(0), 
      f.y.map(_(0)), f.cov.map(_.data(0)))
  }

  out.writeCsv(filtered.map(formatFiltered), 
    rfc.withHeader("time", "state_mean", "state_variance", "one_step_forecast", "one_step_variance"))
}

object SmoothDlm extends App with FirstOrderDlm with SimulatedData {
  val filtered = KalmanFilter.kalmanFilter(mod, data, p)
  val smoothed = Smoothing.backwardsSmoother(mod, p)(filtered)

  val out = new java.io.File("data/FirstOrderDlmSmoothed.csv")

  def formatSmoothed(s: Smoothing.SmoothingState) = 
    (s.time, s.state.mean(0), s.state.covariance.data(0))

  out.writeCsv(smoothed.map(formatSmoothed),
    rfc.withHeader("time", "smoothed_mean", "smoothed_variance"))
}
 
object GibbsParameters extends App with FirstOrderDlm with SimulatedData {
  val iters = gibbsSamples(mod, Gamma(1.0, 10.0), Gamma(1.0, 10.0), p, data).
    steps.
    take(10000)

  val out = new java.io.File("data/FirstOrderDlmGibbs.csv")
  val writer = out.asCsvWriter[(Double, Double, Double, Double)](rfc.withHeader("V", "W", "m0", "c0"))

  def formatParameters(p: Parameters) = {
    (p.v.data(0), p.w.data(0), p.m0.data(0), p.c0.data(0))
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.p))
  }

  writer.close()
}
