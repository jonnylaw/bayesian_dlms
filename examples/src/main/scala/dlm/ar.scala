package examples.dlm

import dlm.core.model._
import Dlm._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions._
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import akka.actor.ActorSystem
import akka.stream._
import scaladsl._

trait ArData {
  val rawData = Paths.get("examples/data/ar_dlm.csv")
  val reader = rawData.asCsvReader[(Double, Double, Double)](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => (a._1, a._2.some)
  }.toVector
}

object SimulateArDlm extends App {
  val p = SvParameters(0.8, 1.0, 0.3)
  def stepDlm(t: Double, x: Double) =
    for {
      x1 <- StochasticVolatility.stepState(p, x)
      y <- Gaussian(x1, math.sqrt(0.5))
    } yield (t + 1.0, y, x1)

  val initVar = math.pow(p.sigmaEta, 2) / (1 - math.pow(p.phi, 2))
  val initState = Gaussian(p.mu, math.sqrt(initVar)).draw
  val init = (0.0, initState, 0.0)
  val sims = MarkovChain(init) { case (t, y, x) => stepDlm(t, x) }.steps
    .take(5000)

  val out = new java.io.File("examples/data/ar_dlm.csv")
  val headers = rfc.withHeader("time", "observation", "state")
  val writer = out.asCsvWriter[(Double, Double, Double)](headers)

  while (sims.hasNext) {
    writer.write(sims.next)
  }

  writer.close()
}

object FilterArDlm extends App with ArData {
  val p = SvParameters(0.8, 1.0, 0.3)
  val filtered = FilterAr.filterUnivariate(data, Vector.fill(data.size)(0.5), p)

  val out = new java.io.File("examples/data/ar_dlm_filtered.csv")

  def formatFiltered(f: FilterAr.FilterState) = {
    (f.time, f.mt, f.ct)
  }
  val headers =
    rfc.withHeader("time", "state_mean", "state_variance")

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object ParametersAr extends App with ArData {
  implicit val system = ActorSystem("ar_dlm")
  implicit val materializer = ActorMaterializer()

  import StochasticVolatility._
  import StochasticVolatilityKnots._

  val p = SvParameters(0.2, 1.0, 0.3)
  val priorMu = Gaussian(1.0, 1.0)
  val priorPhi = Gaussian(0.8, 0.1)
  val priorSigma = InverseGamma(5.0, 1.0)
  val priorV = InverseGamma(5.0, 20.0)
  val f = (dt: Double) => DenseMatrix(1.0)

  val step = (s: (StochVolState, DenseMatrix[Double])) =>
    for {
      theta <- FilterAr.ffbs(s._1.params,
                             data,
                             Vector.fill(data.size)(s._2(0, 0)))
      st = theta.map(x => (x.time, x.sample))
      phi <- samplePhiConjugate(priorPhi, s._1.params, st.map(_._2))
      mu <- sampleMu(priorMu, s._1.params.copy(phi = phi), st.map(_._2))
      sigma <- sampleSigma(priorSigma,
                           s._1.params.copy(mu = mu, phi = phi),
                           st.map(_._2))
      v <- GibbsSampling.sampleObservationMatrix(
        priorV,
        f,
        data.map(x => DenseVector(x._2)),
        st.map { case (t, x) => (t, DenseVector(x)) })
    } yield (StochVolState(SvParameters(phi, mu, sigma), theta), v)

  val initState = FilterAr.ffbs(p, data, Vector.fill(data.size)(priorV.draw))
  val init = (StochVolState(p, initState.draw), DenseMatrix(priorV.draw))

  val iters = MarkovChain(init)(step)

  def formatParameters(s: (StochVolState, DenseMatrix[Double])) =
    List(s._2(0, 0), s._1.params.phi, s._1.params.mu, s._1.params.sigmaEta)

  Streaming
    .writeParallelChain(iters,
                        2,
                        10000,
                        "examples/data/ar_dlm_params",
                        formatParameters)
    .runWith(Sink.onComplete(_ => system.terminate()))
}

object SimulateOuDlm extends App {
  val p = SvParameters(0.2, 1.0, 0.3)
  def stepDlm(t: Double, dt: Double, x: Double) =
    for {
      x1 <- StochasticVolatility.stepOu(p, x, dt)
      y <- Gaussian(x1, math.sqrt(0.5))
    } yield (t + dt, y, x1)
  val deltas = Vector.fill(5000)(scala.util.Random.nextDouble())

  val init = Gaussian(p.mu, math.sqrt(p.sigmaEta * p.sigmaEta / 2 * p.phi))
  val sims = deltas.scanLeft((0.0, 0.0, init.draw)) {
    case ((t, y, xt), dt) =>
      stepDlm(t + dt, dt, xt).draw
  }

  val out = new java.io.File("examples/data/ou_dlm.csv")
  val headers = rfc.withHeader("time", "y", "x")
  out.writeCsv(sims, headers)
}

object FilterOuDlm extends App {
  val rawData = Paths.get("examples/data/ou_dlm.csv")
  val reader = rawData.asCsvReader[(Double, Double, Double)](rfc.withHeader)
  val ys = reader.collect {
    case Right(a) => (a._1, a._2.some)
  }.toVector

  val p = SvParameters(0.2, 1.0, 0.3)

  val filtered = FilterOu.filterUnivariate(ys, Vector.fill(ys.size)(0.5), p)

  val out = new java.io.File("examples/data/ou_dlm_filtered.csv")

  def formatFiltered(f: FilterAr.FilterState) = {
    (f.time, f.mt, f.ct)
  }
  val headers =
    rfc.withHeader("time", "state_mean", "state_variance")

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object FitOuDlm extends App {
  implicit val system = ActorSystem("ou_dlm")
  implicit val materializer = ActorMaterializer()

  import StochasticVolatility._
  val rawData = Paths.get("examples/data/ou_dlm.csv")
  val reader = rawData.asCsvReader[(Double, Double, Double)](rfc.withHeader)
  val ys = reader
    .collect {
      case Right(a) => (a._1, a._2.some)
    }
    .toVector


  val p = SvParameters(0.2, 1.0, 0.3)
  val priorPhi = new Beta(2.0, 5.0)
  val priorMu = Gaussian(1.0, 1.0)
  val priorSigma = InverseGamma(5.0, 1.0)
  val priorV = InverseGamma(2.0, 2.0)
  val f = (dt: Double) => DenseMatrix(1.0)

  val step = (s: (StochasticVolatilityKnots.OuSvState, DenseMatrix[Double])) =>
    for {
      theta <- FilterOu.ffbs(s._1.params, ys, Vector.fill(ys.size)(s._2(0, 0)))
      st = theta.map(x => (x.time, x.sample))
      (phi, acceptedPhi) <- samplePhiOu(priorPhi, s._1.params, st, 0.05, 0.25)(
        s._1.params.phi)
      (mu, acceptedMu) <- sampleMuOu(priorMu, 0.2, s._1.params.copy(phi = phi), st)(
        s._1.params.mu)
      (sigma, acceptedSigma) <- sampleSigmaMetropOu(priorSigma,
                                                    0.1,
                                                    s._1.params.copy(phi = phi, mu = mu),
                                                    st)(s._1.params.sigmaEta)
      v <- GibbsSampling.sampleObservationMatrix(
        priorV,
        f,
        ys.map(x => DenseVector(x._2)),
        st.map { case (t, x) => (t, DenseVector(x)) })
      accepted = DenseVector(Array(acceptedPhi, acceptedMu, acceptedSigma))
    } yield
      (StochasticVolatilityKnots.OuSvState(SvParameters(phi, mu, sigma),
                                           theta,
                                           s._1.accepted + accepted),
       v)
  val initState = FilterOu.ffbs(p, ys, Vector.fill(ys.size)(priorV.draw))
  val init = (StochasticVolatilityKnots.OuSvState(p,
                                                  initState.draw,
                                                  DenseVector.zeros[Int](3)),
              DenseMatrix(priorV.draw))

  val iters = MarkovChain(init)(step)

  def formatParameters(
      s: (StochasticVolatilityKnots.OuSvState, DenseMatrix[Double])) =
    List(s._2(0, 0), s._1.params.phi, s._1.params.mu, s._1.params.sigmaEta)

  Streaming
    .writeParallelChain(iters,
                        2,
                        100000,
                        "examples/data/ou_dlm_params",
                        formatParameters)
    .runWith(Sink.onComplete(_ => system.terminate()))
}
