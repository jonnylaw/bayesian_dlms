package com.github.jonnylaw.dlm.example

import com.github.jonnylaw.dlm._
import Dlm._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.{RandBasis, Gamma, Rand}
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import plot._
import com.cibo.evilplot.plot.aesthetics.DefaultTheme._
import math.exp

trait FirstOrderDlm {
  val mod = Dlm(
    f = (t: Double) => DenseMatrix((1.0)),
    g = (t: Double) => DenseMatrix((1.0))
  )
  val p = DlmParameters(v = DenseMatrix(2.0),
                        w = DenseMatrix(3.0),
                        DenseVector(0.0),
                        DenseMatrix(1.0))

  // particles in particle filter
  val n = 1000
  // if neff < n0 then resample
  val n0 = 250
}

trait SimulatedData {
  val rawData = Paths.get("examples/data/first_order_dlm.csv")
  val reader = rawData.asCsvReader[(Double, Double, Double)](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Data(a._1, DenseVector(a._2.some))
  }.toVector
}

object SimulateDlm extends App with FirstOrderDlm {
  val sims = simulateRegular(mod, p, 1.0).steps.take(1000).toVector

  val out = new java.io.File("examples/data/first_order_dlm.csv")
  val headers = rfc.withHeader("time", "observation", "state")

  def formatData(d: (Data, DenseVector[Double])) = d match {
    case (Data(t, y), x) =>
      t :: KalmanFilter.flattenObs(y).data.toList ::: x.data.toList
  }
  out.writeCsv(sims.map(formatData), headers)
}

object FilterDlm extends App with SimulatedData {
  val mod = Dlm.polynomial(1)
  val p = DlmParameters(v = DenseMatrix(2.0),
                        w = DenseMatrix(3.0),
                        DenseVector(0.0),
                        DenseMatrix(10.0))

  val filtered =
    KalmanFilter(KalmanFilter.advanceState(p, mod.g)).filter(mod, data, p)

  val out = new java.io.File("examples/data/first_order_dlm_filtered.csv")

  def formatFiltered(s: KfState) = s.ft match {
    case Some(f) =>
      List(s.time, s.mt(0), s.ct(0, 0), f(0), s.qt.get(0, 0))
    case _ =>
      List(s.time, s.mt(0), s.ct(0, 0))
  }
  val headers = rfc.withHeader("time",
                               "state_mean",
                               "state_variance",
                               "one_step_forecast",
                               "one_step_variance")

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object AuxFilterFo extends App with FirstOrderDlm with SimulatedData {
  val filtered = AuxFilter(n).filter(mod, data, p)

  val out = new java.io.File("examples/data/fodlm_aux_filtered.csv")

  def formatFiltered(f: PfState) = {
    List(f.time) ++ LiuAndWestFilter.meanState(f.state).data.toList ++
      LiuAndWestFilter.varState(f.state).data.toList
  }
  val headers = rfc.withHeader("time", "state_mean", "state_variance")

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object ParticleFilterFo extends App with FirstOrderDlm with SimulatedData {

  val filtered =
    ParticleFilter(n,
                   math.floor(n / 5).toInt,
                   ParticleFilter.multinomialResample).filter(mod, data, p)

  val out = new java.io.File("examples/data/fodlm_pf.csv")

  def formatFiltered(f: PfState) = {
    List(f.time) ++ LiuAndWestFilter.meanState(f.state).data.toList ++
      LiuAndWestFilter.varState(f.state).data.toList
  }
  val headers = rfc.withHeader("time", "state_mean", "state_variance")

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object LiuAndWest extends App with FirstOrderDlm with SimulatedData {
  // smoothing parameter for the mixture of gaussians,
  // equal to (3 delta - 1) / 2 delta
  val a = (3 * 0.95 - 1) / 2 * 0.95

  val prior = for {
    v <- InverseGamma(3.0, 4.0)
    w <- InverseGamma(3.0, 3.0)
  } yield DlmParameters(DenseMatrix(v), DenseMatrix(w), p.m0, p.c0)

  val filtered = LiuAndWestFilter(n, prior, a, n).filter(mod, data, p)

  val out = new java.io.File("examples/data/liuandwest_filtered.csv")

  def formatFiltered(s: PfStateParams): List[Double] = {
    List(s.time) ++ LiuAndWestFilter.meanState(s.state).data.toList ++
      LiuAndWestFilter.meanParameters(s.params map (_.map(exp))).toList ++
      LiuAndWestFilter.varParameters(s.params map (_.map(exp))).data.toList
  }
  val headers = rfc.withHeader("time",
                               "state_mean",
                               "v_mean",
                               "w_mean",
                               "m0_mean",
                               "c0_mean",
                               "v_variance",
                               "w_variance",
                               "m0_variance",
                               "c0_variance")

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object ConjFilter extends App with FirstOrderDlm with SimulatedData {
  override val p = DlmParameters(v = DenseMatrix(2.0),
                                 w = DenseMatrix(3.0),
                                 m0 = DenseVector(0.0),
                                 c0 = DenseMatrix(100.0))

  val prior = InverseGamma(3.0, 4.0)
  val filtered = ConjugateFilter(prior, ConjugateFilter.advanceState(p, mod.g))
    .filter(mod, data, p)

  val out =
    new java.io.File("examples/data/first_order_dlm_conjugate_filtered.csv")

  def formatFiltered(s: InverseGammaState) = {
    List(s.kfState.time,
         s.kfState.mt(0),
         s.kfState.ct(0, 0),
         s.variance.head.mean,
         s.variance.head.variance)
  }

  val headers = rfc.withHeader("time",
                               "state_mean",
                               "state_variance",
                               "v_mean",
                               "v_variance")

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object Storvik extends App with FirstOrderDlm with SimulatedData {
  val priorV = InverseGamma(3.0, 4.0)
  val priorW = InverseGamma(3.0, 10.0)

  val filtered = StorvikFilter.filterTs(mod, data, p, priorV, priorW, n, n)

  val out = new java.io.File("examples/data/fo_storvik_filtered.csv")

  def formatFiltered(s: StorvikState): List[Double] = {
    List(s.time) ++ LiuAndWestFilter.meanState(s.state).data.toList ++
      LiuAndWestFilter.meanParameters(s.params).toList ++
      LiuAndWestFilter.varParameters(s.params).data.toList
  }

  val headers = rfc.withHeader("time",
                               "state_mean",
                               "v_mean",
                               "w_mean",
                               "m0_mean",
                               "c0_mean",
                               "v_variance",
                               "w_variance",
                               "m0_variance",
                               "c0_variance")

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object RbFilter extends App with FirstOrderDlm with SimulatedData {
  // smoothing parameter for the mixture of gaussians,
  // equal to (3 delta - 1) / 2 delta
  val delta = 0.99
  val a = (3 * delta - 1) / 2 * delta

  val prior = for {
    v <- InverseGamma(3.0, 4.0)
    w <- InverseGamma(3.0, 10.0)
  } yield DlmParameters(DenseMatrix(v), DenseMatrix(w), p.m0, p.c0)

  val filtered = RaoBlackwellFilter(500, prior, a, 250).filter(mod, data, p)

  val out = new java.io.File("examples/data/fo_raoblackwellfilter.csv")

  def formatFiltered(s: RbState): List[Double] = {
    List(s.time) ++ LiuAndWestFilter.meanState(s.mt).data.toList ++
      LiuAndWestFilter.meanParameters(s.params map (_.map(exp))).toList ++
      LiuAndWestFilter.varParameters(s.params map (_.map(exp))).data.toList
  }

  val headers = rfc.withHeader("time",
                               "state_mean",
                               "v_mean",
                               "w_mean",
                               "m0_mean",
                               "c0_mean",
                               "v_variance",
                               "w_variance",
                               "m0_variance",
                               "c0_variance")

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object SmoothDlm extends App with SimulatedData {
  val mod = Dlm.polynomial(1)
  val p = DlmParameters(v = DenseMatrix(2.0),
                        w = DenseMatrix(3.0),
                        DenseVector(0.0),
                        DenseMatrix(10.0))

  val filtered =
    KalmanFilter(KalmanFilter.advanceState(p, mod.g)).filter(mod, data, p)
  val smoothed = Smoothing.backwardsSmoother(mod)(filtered)

  val out = new java.io.File("examples/data/first_order_dlm_smoothed.csv")

  def formatSmoothed(s: Smoothing.SmoothingState) =
    (s.time, s.mean(0), s.covariance.data(0))

  out.writeCsv(smoothed.map(formatSmoothed),
               rfc.withHeader("time", "smoothed_mean", "smoothed_variance"))
}

object SampleStates extends App with FirstOrderDlm with SimulatedData {
  implicit val basis = RandBasis.withSeed(7)

  val sampled = Smoothing.ffbsDlm(mod, data, p).sample(1000)
  val meanState = SvdSampler.meanState(sampled)
  val out = new java.io.File("examples/data/first_order_state_sample.csv")

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
                       "examples/data/first_order_dlm_gibbs.csv",
                       headers)(iters)
}
