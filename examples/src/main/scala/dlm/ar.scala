package examples.dlm

import dlm.core.model._
import Dlm._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.{MarkovChain, Beta, Gaussian}
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._

trait ArDlm {
  val mod = Dlm.autoregressive(phi = 0.9)
  val p = DlmParameters(
    DenseMatrix(4.0),
    DenseMatrix(2.0),
    DenseVector(0.0),
    DenseMatrix(1.0))
}

trait ArData {
  val rawData = Paths.get("examples/data/ar_dlm.csv")
  val reader = rawData.asCsvReader[(Double, Double, Double)](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Data(a._1, DenseVector(a._2.some))
  }.toVector
}

object SimulateArDlm extends App with ArDlm {
  val sims = simulateRegular(mod, p, 1.0).steps.take(1000)

  val out = new java.io.File("examples/data/ar_dlm.csv")
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

object FilterArDlm extends App with ArDlm with ArData {
  val filtered = SvdFilter.filterDlm(mod, data, p)

  val out = new java.io.File("examples/data/ar_dlm_filtered.csv")

  def formatFiltered(f: SvdState) = {
    val ct = f.uc * diag(f.dc) * f.uc.t
    (f.time, f.mt(0), ct(0, 0), f.ft(0))
  }
  val headers =
    rfc.withHeader("time", "state_mean", "state_variance", "one_step_forecast")

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object ParametersAr extends App with ArDlm with ArData {
  val priorV = InverseGamma(5.0, 20.0)
  val priorW = InverseGamma(6.0, 10.0)
  val priorPhi = new Beta(20, 2)

  val prior = for {
    v <- priorV
    w <- priorW
  } yield DlmParameters(DenseMatrix(v), DenseMatrix(w), p.m0, p.c0)

  val step = (s: (Double, GibbsSampling.State)) =>
    for {
      newS <- GibbsSampling.dinvGammaStep(
        GibbsSampling.updateModel(mod, s._1),
        priorV,
        priorW,
        data)(s._2)
      phi <- GibbsSampling.samplePhi(priorPhi, 1000, 0.5, newS)(s._1)
    } yield (phi, newS)

  val init = for {
    p <- prior
    phi <- priorPhi
    state <- Smoothing.ffbsDlm(mod, data, p)
  } yield (phi, GibbsSampling.State(p, state))


  val iters = MarkovChain(init.draw)(step).steps.take(100000)

  val out = new java.io.File("examples/data/ar_dlm_gibbs.csv")
  val writer = out.asCsvWriter[List[Double]](rfc.withHeader(false))

  def formatParameters(s: (Double, GibbsSampling.State)) = {
    s._1 :: s._2.p.v.data(0) :: s._2.p.w.data(0) :: Nil
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next))
  }

  writer.close()
}

object SimulateOuDlm extends App {
  val p = SvParameters(0.2, 1.0, 0.3)
  def stepDlm(t: Double, dt: Double, x: Double) = for {
    x1 <- StochasticVolatility.stepOu(p, x, dt)
    y <- Gaussian(x1, 1.0)
  } yield (t + dt, y, x1)
  val deltas = Vector.fill(1000)(scala.util.Random.nextDouble())

  val init = Gaussian(p.mu, math.sqrt(p.sigmaEta * p.sigmaEta / p.phi * p.phi))
  val sims = deltas.scanLeft((0.0, 0.0, init.draw)){ case ((t, y, xt), dt) =>
    stepDlm(t + dt, dt, xt).draw }

  val out = new java.io.File("examples/data/ou_dlm.csv")
  val headers = rfc.withHeader("time", "y", "x")
  out.writeCsv(sims, headers)
}

object FitOuDlm extends App {
  val rawData = Paths.get("examples/data/ar_dlm.csv")
  val reader = rawData.asCsvReader[(Double, Double, Double)](rfc.withHeader)
  val ys = reader.collect {
    case Right(a) => (a._1, a._2.some)
  }.toVector

  val p = SvParameters(0.2, 1.0, 0.3)
  val priorPhi = new Beta(2.0, 5.0)
  val priorMu = Gaussian(1.0, 1.0)
  val priorSigma = InverseGamma(10.0, 1.0)
  val priorV = InverseGamma(2.0, 2.0)
  val f = (dt: Double) => DenseMatrix(1.0)
  val v = 1.0

  val step = (s: StochVolState) => for {
    theta <- FilterOu.ffbs(p, ys, Vector.fill(ys.size)(v))
    st = theta.map(x => (x.time, x.sample))
    (phi, _) <- StochasticVolatility.samplePhiOu(priorPhi,
      s.params, st, 0.05, 0.5)(s.params.phi)
    (mu, _) <- StochasticVolatility.sampleMuOu(priorMu, 0.2,
      s.params, st)(s.params.mu)
    (sigma, _) <- StochasticVolatility.sampleSigmaMetropOu(priorSigma,
      0.05, p, st)(s.params.sigmaEta)
    // v <- GibbsSampling.sampleObservationMatrix(priorV, f,
    //   data.map(x => Data(x._1, DenseVector(x._2))), theta)
  } yield StochVolState(SvParameters(phi, mu, sigma), theta, 0)

  val initState = FilterOu.ffbs(p, ys, Vector.fill(ys.size)(v))
  val init = StochVolState(p, initState.draw, 0)
  val iters = MarkovChain(init)(step).steps.take(10000)

  val out = new java.io.File("examples/data/ou_dlm_params.csv")
  val writer = out.asCsvWriter[List[Double]](rfc.withHeader("phi", "mu", "sigma"))

  def formatParameters(s: StochVolState) = 
    List(s.params.phi, s.params.mu, s.params.sigmaEta)

  while (iters.hasNext) {
    writer.write(formatParameters(iters.next))
  }

  writer.close()
}
