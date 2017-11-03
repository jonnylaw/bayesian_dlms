package dlm.examples

import dlm.model._
import Dlm._
import GibbsSampling._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.{Gamma, Gaussian, Rand, RandBasis}
import breeze.numerics.exp
import breeze.stats.mean
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

/**
  * Simulate data from a Seasonal DLM
  */
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

/**
  * Filter the seasonal DLM
  */
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

/**
  * Run backward smoothing on the seasonal DLM
  */
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

/**
  * Use Gibbs sampling with Inverse Gamma priors on the observation variance and diagonal system covariance
  */
object SeasonalGibbsSampling extends App with SeasonalModel with SeasonalData {
  implicit val basis = RandBasis.withSeed(7)

  val iters = GibbsSampling.gibbsSamples(mod, InverseGamma(5.0, 4.0), 
    InverseGamma(17.0, 4.0), p, data).
    steps.
    take(10000)

  val headers = rfc.withHeader("V", "W1", "W2", "W3", "W4", "W5", "W6", "W7")

  def formatParameters(p: Parameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  Streaming.writeChain(formatParameters, 
    "data/seasonal_dlm_gibbs.csv", headers)(iters.map(_.p))
}

object ForecastSeasonal extends App with SeasonalModel with SeasonalData {
  // read in the parameters from the MCMC chain and caculate the mean
  val mcmcChain = Paths.get("data/seasonal_dlm_gibbs.csv")
  val read = mcmcChain.asCsvReader[List[Double]](rfc.withHeader)

  val params: List[Double] = read.
    collect { case Success(a) => a }.
    toList.
    transpose.
    map(a => mean(a))

  val meanParameters = Parameters(
    DenseMatrix(params.head), 
    diag(DenseVector(params.tail.toArray)), 
    p.m0,
    p.c0)

  // get the posterior distribution of the final state
  val filtered = KalmanFilter.kalmanFilter(mod, data, meanParameters)
  val (mt, ct, initTime) = filtered.map(a => (a.mt, a.ct, a.time)).last
  
  val forecasted = Dlm.forecast(mod, mt, ct, initTime, meanParameters).
    take(100).
    toList

  val out = new java.io.File("data/seasonal_model_forecast.csv")
  val headers = rfc.withHeader("Time", "Observation", "Variance")
  val writer = out.writeCsv(forecasted, headers)
}

/**
  * Sample the state using FFBS algorithm
  */
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

/**
  * Using the Metropolis alogrithm to determine the parameters of the simulated Seasonal Model
  */
object SeasonalMetropolis extends App with SeasonalModel with SeasonalData {
  val prior = (p: Parameters) => 0.0

  val iters = Metropolis.dlm(mod, data, 
    Metropolis.symmetricProposal(1e-2), prior, p).
    steps.
    take(100000)

  val out = new java.io.File("data/seasonal_dlm_metropolis.csv")
  val headers = rfc.withHeader("V", "W1", "W2", "W3", "W4", "W5", "W6", "W7")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatParameters(p: Parameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.parameters))
  }

  writer.close()
}
