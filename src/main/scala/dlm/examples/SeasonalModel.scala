package dlm.examples

import dlm.model._
import Dlm._
import GibbsSampling._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.Gamma
import cats.implicits._
import java.nio.file.Paths
import kantan.csv._
import kantan.csv.ops._
import kantan.csv.generic._

trait SeasonalModel {
  val mod = Dlm.polynomial(1) |+| Dlm.seasonal(24, 3)

  val p = Dlm.Parameters(
    v = DenseMatrix((1.906964895e-18)),
    w = diag(DenseVector(
      1.183656514e-01,
      4.074187352e-13,
      4.162722546e-01, 
      7.306081843e-14, 
      2.193511341e-03, 
      1.669158400e-08, 
      3.555685730e-03)),
    m0 = DenseVector.fill(7)(0.0),
    c0 = diag(DenseVector.fill(7)(1.0))
  )
}

trait SeasonalData {
  val rawData = Paths.get("data/seasonal_dlm.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.
    collect { 
      case Success(a) => Data(a.head.toInt, Some(a(1)).map(DenseVector(_)))
    }.
    toArray
}

object SimulateSeasonalDlm extends App with SeasonalModel {
  val sims = simulate(0, mod, p).
    steps.
    take(1000)

  val out = new java.io.File("data/seasonal_dlm.csv")
  val headers = rfc.withHeader("time", "observation", "state1", "state2", "state3", "state4", "state5", "state6", "state7")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatData(d: (Data, DenseVector[Double])) = d match {
    case (Data(t, y), x) =>
      List(t.toDouble) ++ y.get.data.toList ++ x.data.toList
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

object FilterSeasonalDlm extends App with SeasonalModel with SeasonalData {
  val filtered = KalmanFilter.kalmanFilter(mod, data, p)

  val out = new java.io.File("data/seasonal_filtered.csv")

  def formatFiltered(f: KalmanFilter.State) = {
    (f.time, DenseVector.vertcat(f.mt, diag(f.ct)).data.toList)
  }
  val headers = rfc.withHeader("time", "state_mean_1", "state_mean_2", "state_mean_3", "state_mean_4", "state_mean_5", "state_mean_6", "state_mean_7", 
    "state_variance_1", "state_variance_2", "state_variance_3", "state_variance_4", "state_variance_5", "state_variance_6", "state_variance_7")

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object SmoothSeasonalDlm extends App with SeasonalModel with SeasonalData {
  val filtered = KalmanFilter.kalmanFilter(mod, data, p)
  val smoothed = Smoothing.backwardsSmoother(mod, p)(filtered)

  val out = new java.io.File("data/seasonal_smoothed.csv")

  def formatSmoothed(s: Smoothing.SmoothingState) = 
    (s.time, DenseVector.vertcat(s.mean, diag(s.covariance)).data.toList)

  val headers = rfc.withHeader("time", "state_mean_1", "state_mean_2", "state_mean_3", "state_mean_4", "state_mean_5", "state_mean_6", "state_mean_7", 
    "state_variance_1", "state_variance_2", "state_variance_3", "state_variance_4", "state_variance_5", "state_variance_6", "state_variance_7")

  out.writeCsv(smoothed.map(formatSmoothed), headers)
}

object SeasonalGibbsSampling extends App with SeasonalModel with SeasonalData {
  val iters = gibbsSamples(mod, Gamma(1.0, 10.0), Gamma(1.0, 10.0), p, data).
    steps.
    take(10000)

  val out = new java.io.File("data/seasonal_dlm_gibbs.csv")
  val writer = out.asCsvWriter[List[Double]](rfc.withHeader("V", "W1", "W2", "W3", "W4", "W5", "W6", "W7"))

  def formatParameters(p: Parameters) = {
    (p.v.data ++ p.w.data).toList
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.p))
  }

  writer.close()
}
