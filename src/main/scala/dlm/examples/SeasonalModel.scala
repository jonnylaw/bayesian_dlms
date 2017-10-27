package dlm.examples

import dlm.model._
import Dlm._
import GibbsSampling._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.{Gamma, Gaussian, Rand}
import breeze.numerics.exp
import cats.Applicative
import cats.implicits._
import java.nio.file.Paths
import kantan.csv._
import kantan.csv.ops._

trait SeasonalModel {
  val mod = Dlm.polynomial(1) |+| Dlm.seasonal(24, 3)

  val p = Dlm.Parameters(
    v = DenseMatrix((1.0)),
    w = diag(DenseVector(0.01, 0.2, 0.4, 0.5, 0.2, 0.1, 0.4)),
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
    f.time.toDouble +: DenseVector.vertcat(f.mt, diag(f.ct)).data.toList
  }

  val headers = rfc.withHeader("time", "state_mean_1", "state_mean_2", "state_mean_3", "state_mean_4", 
    "state_mean_5", "state_mean_6", "state_mean_7",
    "state_variance_1", "state_variance_2", "state_variance_3", 
    "state_variance_4", "state_variance_5", "state_variance_6", "state_variance_7")

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object SmoothSeasonalDlm extends App with SeasonalModel with SeasonalData {
  val filtered = KalmanFilter.kalmanFilter(mod, data, p)
  val smoothed = Smoothing.backwardsSmoother(mod, p)(filtered)

  val out = new java.io.File("data/seasonal_smoothed.csv")

  def formatSmoothed(s: Smoothing.SmoothingState) = 
    s.time.toDouble +: DenseVector.vertcat(s.mean, diag(s.covariance)).data.toList

  val headers = rfc.withHeader("time", "state_mean_1", "state_mean_2", 
    "state_mean_3", "state_mean_4", "state_mean_5", "state_mean_6", "state_mean_7",
    "state_variance_1", "state_variance_2", "state_variance_3", "state_variance_4", 
    "state_variance_5", "state_variance_6", "state_variance_7")

  out.writeCsv(smoothed.map(formatSmoothed), headers)
}

object SeasonalGibbsSampling extends App with SeasonalModel with SeasonalData {
  val iters = GibbsSampling.gibbsSamples(mod, InverseGamma(5.0, 4.0), InverseGamma(17.0, 4.0), p, data).
    steps.
    take(10000)

  val out = new java.io.File("data/seasonal_dlm_gibbs.csv")
  val writer = out.asCsvWriter[List[Double]](rfc.withHeader("V", "W1", "W2", "W3", "W4", "W5", "W6", "W7"))

  def formatParameters(p: Parameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.p))
  }

  writer.close()
}

object SampleStates extends App with SeasonalModel with SeasonalData {
  val iters = Iterator.fill(10000)(GibbsSampling.sampleState(mod, data, p))

  val out = new java.io.File("data/seasonal_dlm_state_2_samples.csv")
  val writer = out.asCsvWriter[List[Double]](rfc.withoutHeader)

  /**
    * Select only the nth state
    */
  def formatState(n: Int)(s: Array[(Time, DenseVector[Double])]) = {
    val state: List[List[Double]] = s.map(_._2.data.toList).toList.transpose
    state(n)
  }

  val headers = rfc.withHeader(false)

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatState(1)(iters.next))
  }

  writer.close()
}

object SeasonalMetropolisHastings extends App with SeasonalModel with SeasonalData {
  def proposal(delta: Double) = { p: Parameters =>
    for {
      m <- p.m0.data.toVector traverse (x => Gaussian(x, delta): Rand[Double])
      innovc <- Applicative[Rand].replicateA(7, Gaussian(0, delta))
      c = diag(p.c0) *:* DenseVector(exp(innovc.toArray))
      innovw <- Applicative[Rand].replicateA(7, Gaussian(0, delta))
      w = diag(p.w) *:* DenseVector(exp(innovw.toArray))
      innovv <- Gaussian(0, delta)
      v = diag(p.v) *:* DenseVector(exp(innovv))
    } yield Parameters(diag(v), diag(w), DenseVector(m.toArray), diag(c))
  }

  val iters = MetropolisHastings.metropolisHastingsDlm(mod, data, proposal(1e-2), p).
    steps.
    take(100000)

  val out = new java.io.File("data/seasonal_dlm_metropolis.csv")
  val writer = out.asCsvWriter[List[Double]](rfc.withHeader("V", "W1", "W2", "W3", "W4", "W5", "W6", "W7"))

  def formatParameters(p: Parameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.parameters))
  }

  writer.close()
}
