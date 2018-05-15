package dlm.examples

import dlm.model._
import Dlm._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.{MarkovChain, Beta}
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._

trait ArDlm {
  val mod = Dlm.autoregressive(phi = 0.9)
  val p = Dlm.Parameters(
    DenseMatrix(4.0), 
    DenseMatrix(2.0), 
    DenseVector(0.0), 
    DenseMatrix(1.0))
}

trait ArData {
  val rawData = Paths.get("data/ar_dlm.csv")
  val reader = rawData.asCsvReader[(Double, Double, Double)](rfc.withHeader)
  val data = reader.
    collect { 
      case Right(a) => Data(a._1, DenseVector(a._2.some))
    }.
    toVector
}

object SimulateArDlm extends App with ArDlm {
  val sims = simulateRegular(0, mod, p, 1.0).
    steps.
    take(1000)

  val out = new java.io.File("data/ar_dlm.csv")
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
  val filtered = SvdFilter.filter(mod, data, p)

  val out = new java.io.File("data/ar_dlm_filtered.csv")

  def formatFiltered(f: SvdFilter.State) = {
    val ct = f.uc * diag(f.dc) * f.uc.t
    (f.time, f.mt(0), ct(0, 0), f.ft(0))
  }
  val headers = rfc.withHeader("time", "state_mean", "state_variance",
    "one_step_forecast")

  out.writeCsv(filtered.map(formatFiltered), headers)
}
 
object ParametersAr extends App with ArDlm with ArData {
  val priorV = InverseGamma(5.0, 20.0)
  val priorW = InverseGamma(6.0, 10.0)
  val priorPhi = new Beta(20, 2)

  val prior = for {
    v <- priorV
    w <- priorW
  } yield Dlm.Parameters(DenseMatrix(v), DenseMatrix(w), p.m0, p.c0)

  val step = (s: (Double, GibbsSampling.State)) => for {
    newS <- GibbsSampling.dinvGammaStep(GibbsSampling.updateModel(mod, s._1),
      priorV, priorW, data)(s._2)
    phi <- GibbsSampling.samplePhi(priorPhi, 1000, 0.5, newS)(s._1)
  } yield (phi, newS)

  val init = for {
    p <- prior
    phi <- priorPhi
    state <- Smoothing.ffbs(mod, data, p)
  } yield (phi, GibbsSampling.State(p, state.toArray))

  val iters = MarkovChain(init.draw)(step).steps.take(100000)

  val out = new java.io.File("data/ar_dlm_gibbs.csv")
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
