package examples.dlm

import core.dlm.model._
import Dlm._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.RandBasis
import breeze.stats.mean
import cats.implicits._
import java.nio.file.Paths
import kantan.csv._
import kantan.csv.ops._

trait SeasonalModel {
  val mod = Dlm.polynomial(1) |+| Dlm.seasonal(24, 3)

  val p = DlmParameters(
    v = DenseMatrix((1.0)),
    w = diag(DenseVector(0.01, 0.2, 0.4, 0.5, 0.2, 0.1, 0.4)),
    m0 = DenseVector.fill(7)(0.0),
    c0 = diag(DenseVector.fill(7)(1.0))
  )
}

trait SeasonalData {
  val rawData = Paths.get("examples/data/seasonal_dlm.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Data(a.head, DenseVector(a(1).some))
  }.toVector
}

/**
  * Simulate data from a Seasonal DLM
  */
object SimulateSeasonalDlm extends App with SeasonalModel {
  val sims = simulateRegular(mod, p, 1.0).steps.take(1000)

  val out = new java.io.File("examples/data/seasonal_dlm.csv")
  val headers = rfc.withHeader("time",
                               "observation",
                               "state1",
                               "state2",
                               "state3",
                               "state4",
                               "state5",
                               "state6",
                               "state7")
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

/**
  * Filter the seasonal DLM
  */
object FilterSeasonalDlm extends App with SeasonalModel with SeasonalData {
  val filtered = SvdFilter(SvdFilter.advanceState(p, mod.g)).filter(mod, data, p)

  val out = new java.io.File("examples/data/seasonal_filtered.csv")

  def formatFiltered(f: SvdState) = {
    val ct = f.uc * diag(f.dc) * f.uc.t
    f.time.toDouble +: DenseVector.vertcat(f.mt, diag(ct)).data.toList
  }

  val hs = Seq("time") ++ (1 to 7).map(i => s"state_mean_$i") ++
    (1 to 7).map(i => s"state_variance_$i")
  val headers = rfc.withHeader(hs: _*)

  out.writeCsv(filtered.map(formatFiltered), headers)
}

/**
  * Run backward smoothing on the seasonal DLM
  */
object SmoothSeasonalDlm extends App with SeasonalModel with SeasonalData {
  val filtered = KalmanFilter(KalmanFilter.advanceState(p, mod.g)).filter(mod, data, p)
  val smoothed = Smoothing.backwardsSmoother(mod)(filtered)

  val out = new java.io.File("examples/data/seasonal_smoothed.csv")

  def formatSmoothed(s: Smoothing.SmoothingState) =
    s.time.toDouble +: DenseVector
      .vertcat(s.mean, diag(s.covariance))
      .data
      .toList

  val headers = rfc.withHeader(
    "time",
    "state_mean_1",
    "state_mean_2",
    "state_mean_3",
    "state_mean_4",
    "state_mean_5",
    "state_mean_6",
    "state_mean_7",
    "state_variance_1",
    "state_variance_2",
    "state_variance_3",
    "state_variance_4",
    "state_variance_5",
    "state_variance_6",
    "state_variance_7"
  )

  out.writeCsv(smoothed.map(formatSmoothed), headers)
}

/**
  * Use Gibbs sampling with Inverse Gamma priors on the observation variance and diagonal system covariance
  */
object SeasonalGibbsSampling extends App with SeasonalModel with SeasonalData {
  implicit val basis = RandBasis.withSeed(7)

  val iters = GibbsSampling
    .sample(mod, InverseGamma(5.0, 4.0), InverseGamma(17.0, 4.0), p, data)
    .steps
    .take(10000)

  val headers = rfc.withHeader("V", "W1", "W2", "W3", "W4", "W5", "W6", "W7")

  def formatParameters(p: DlmParameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  Streaming.writeChain(formatParameters,
                       "examples/data/seasonal_dlm_gibbs.csv",
                       headers)(iters.map(_.p))
}

object ForecastSeasonal extends App with SeasonalModel with SeasonalData {
  // read in the parameters from the MCMC chain and caculate the mean
  val mcmcChain = Paths.get("examples/data/seasonal_dlm_gibbs.csv")
  val read = mcmcChain.asCsvReader[List[Double]](rfc.withHeader)

  val params: List[Double] =
    read.collect { case Right(a) => a }.toList.transpose.map(a => mean(a))

  val meanParameters = DlmParameters(DenseMatrix(params.head),
                                     diag(DenseVector(params.tail.toArray)),
                                     p.m0,
                                     p.c0)

  // get the posterior distribution of the final state
  val filtered = SvdFilter(SvdFilter.advanceState(meanParameters, mod.g)).
    filter(mod, data, meanParameters)
  val (mt, ct, initTime) = filtered.map { a =>
    val ct = a.uc * diag(a.dc) * a.uc.t
    (a.mt, ct, a.time)
  }.last

  val forecasted =
    Dlm.forecast(mod, mt, ct, initTime, meanParameters).take(100).toList

  val out = new java.io.File("examples/data/seasonal_model_forecast.csv")
  val headers = rfc.withHeader("time", "forecast", "variance")
  val writer = out.writeCsv(forecasted, headers)
}

/**
  * Sample the state using FFBS algorithm
  */
object SampleStatesSeasonal extends App with SeasonalModel with SeasonalData {
  val sampled = Smoothing.ffbsDlm(mod, data, p).sample(10000)

  val out = new java.io.File("examples/data/seasonal_dlm_state_2_samples.csv")
  val writer = out.asCsvWriter[List[Double]](rfc.withoutHeader)

  /**
    * Select only the nth state
    */
  def formatState(n: Int)(s: Vector[(Double, DenseVector[Double])]) = {
    val state: List[List[Double]] = s.map(_._2.data.toList).toList.transpose
    state(n)
  }

  val headers = rfc.withHeader(false)
  out.writeCsv(sampled.map(formatState(1)), headers)
}
