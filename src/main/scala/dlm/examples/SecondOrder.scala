package dlm.examples

import dlm.model._
import Dlm._
import MetropolisHastings._
import GibbsSampling._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.{Gaussian, Rand}
import math.{log, exp}
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._

trait DlmModel {
  val mod = polynomial(2)

  val p = Parameters(
    DenseMatrix(3.0),
    diag(DenseVector(2.0, 1.0)),
    DenseVector(0.0, 0.0), 
    diag(DenseVector(100.0, 100.0))
  )
}

trait SimulatedSecondOrderData {
  val rawData = Paths.get("data/second_order_dlm.csv")
  val reader = rawData.asCsvReader[(Time, Double, Double, Double)](rfc.withHeader)
  val data = reader.
    collect { 
      case Success(a) => Data(a._1, Some(a._2).map(DenseVector(_)))
    }.
    toArray
}

object SimulateSecondOrderDlm extends App with DlmModel {
  val sims = simulate(0, mod, p).
    steps.
    take(1000)

  val out = new java.io.File("data/second_order_dlm.csv")
  val headers = rfc.withHeader("time", "observation", "state_1", "state_2")
  val writer = out.asCsvWriter[(Time, Option[Double], Double, Double)](headers)

  def formatData(d: (Data, DenseVector[Double])) = d match {
    case (Data(t, y), x) =>
      (t, y.map(a => a(0)), x(0), x(1))
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

object FilterSecondOrderDlm extends App with DlmModel with SimulatedSecondOrderData {
  val filtered = KalmanFilter.kalmanFilter(mod, data, p)

  val out = new java.io.File("data/second_order_dlm_filtered.csv")

  def formatFiltered(f: KalmanFilter.State) = {
    f.time.toDouble +: DenseVector.vertcat(f.mt, diag(f.ct)).data.toList
  }

  val headers = rfc.withHeader("time", "state_mean_1", "state_mean_2", "state_variance_1", "state_variance_2")

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object SmoothSecondOrderDlm extends App with DlmModel with SimulatedSecondOrderData {
  val filtered = KalmanFilter.kalmanFilter(mod, data, p)
  val smoothed = Smoothing.backwardsSmoother(mod, p)(filtered)

  val out = new java.io.File("data/second_order_dlm_smoothed.csv")

  def formatSmoothed(s: Smoothing.SmoothingState) = 
    (s.time, s.mean(0), s.covariance.data(0))

  out.writeCsv(smoothed.map(formatSmoothed),
    rfc.withHeader("time", "smoothed_mean", "smoothed_variance"))
}

object GibbsSecondOrder extends App with DlmModel with SimulatedSecondOrderData {
  val priorV = InverseGamma(4.0, 9.0)
  val priorW = InverseGamma(3.0, 8.0)

  val iters = sample(mod, priorV, priorW, p, data).
    steps.
    take(10000)

  // write iters to file
  val out = new java.io.File("data/second_order_dlm_gibbs.csv")
  val writer = out.asCsvWriter[List[Double]](rfc.withHeader("V", "W1", "W2"))

  def formatParameters(p: Parameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.p))
  }

  writer.close()
}

object GibbsInvestParameters extends App with DlmModel {
  val data: Array[Data] = scala.io.Source.fromFile("data/invest2.dat").
    getLines.
    map(_.split(",")).
    zipWithIndex.
    map { case (x, i) => Data(i + 1960, Some(DenseVector(x(1).toDouble / 1000.0))) }.
    toArray

  val priorV = InverseGamma(4.0, 10.0)
  val priorW = InverseGamma(4.0, 10.0)

  val initP = Parameters(
    v = DenseMatrix(priorV.draw),
    w = diag(DenseVector.fill(2)(priorW.draw)),
    m0 = p.m0,
    c0 = p.c0
  )

  println(s"Initial parameters: $initP")

  val iters = GibbsSampling.sample(mod, priorV, priorW, initP, data).
    steps.
    drop(12000).
    take(12000)

  val out = new java.io.File("data/gibbs_spain_investment.csv")
  val writer = out.asCsvWriter[List[Double]](rfc.withHeader("V", "W1", "W2"))

  def formatParameters(p: Parameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.p))
  }

  writer.close()
}
