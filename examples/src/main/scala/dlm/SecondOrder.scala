package com.github.jonnylaw.dlm.example

import com.github.jonnylaw.dlm._
import GibbsSampling._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._

trait Dlm {
  val mod = Dlm.polynomial(2)

  val p = DlmParameters(
    DenseMatrix(3.0),
    diag(DenseVector(2.0, 1.0)),
    DenseVector(0.0, 0.0),
    diag(DenseVector(100.0, 100.0))
  )
}

trait SimulatedSecondOrderData {
  val rawData = Paths.get("examples/data/second_order_dlm.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Data(a.head, DenseVector(Some(a(1))))
  }.toVector
}

object SimulateSecondOrderDlm extends App with Dlm {
  val sims = Dlm.simulateRegular(mod, p, 1.0).steps.take(1000)

  val out = new java.io.File("examples/data/second_order_dlm.csv")
  val headers = rfc.withHeader("time", "observation", "state_1", "state_2")
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

object FilterSecondOrderDlm extends App with Dlm with SimulatedSecondOrderData {

  val filtered = SvdFilter.filterDlm(mod, data, p)

  val out = new java.io.File("examples/data/second_order_dlm_filtered.csv")

  def formatFiltered(f: SvdState) = {
    val ct = f.uc * (f.dc.t * f.dc) * f.uc.t
    f.time.toDouble +: DenseVector.vertcat(f.mt, diag(ct)).data.toList
  }

  val headers = rfc.withHeader("time",
                               "state_mean_1",
                               "state_mean_2",
                               "state_variance_1",
                               "state_variance_2")

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object SmoothSecondOrderDlm extends App with Dlm with SimulatedSecondOrderData {

  val filtered = KalmanFilter.filterDlm(mod, data, p)
  val smoothed = Smoothing.backwardsSmoother(mod)(filtered)

  val out = new java.io.File("examples/data/second_order_dlm_smoothed.csv")

  def formatSmoothed(s: Smoothing.SmoothingState) =
    (s.time, s.mean(0), s.covariance.data(0))

  out.writeCsv(smoothed.map(formatSmoothed),
               rfc.withHeader("time", "smoothed_mean", "smoothed_variance"))
}

object GibbsSecondOrder extends App with Dlm with SimulatedSecondOrderData {
  val priorV = InverseGamma(4.0, 9.0)
  val priorW = InverseGamma(3.0, 8.0)

  val iters = sample(mod, priorV, priorW, p, data).steps.take(10000)

  // write iters to file
  val out = new java.io.File("examples/data/second_order_dlm_gibbs.csv")
  val writer = out.asCsvWriter[List[Double]](rfc.withHeader("V", "W1", "W2"))

  def formatParameters(p: DlmParameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.p))
  }

  writer.close()
}

object GibbsInvestParameters extends App with Dlm {
  val rawData = Paths.get("examples/data/invest2.dat")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader(false))
  val data = reader
    .collect {
      case Right(a) => Data(0.0, DenseVector(Some(a(1) / 1000.0)))
    }
    .toVector
    .zipWithIndex
    .map {
      case (d, i) =>
        d.copy(time = i + 1960.0)
    }

  val priorV = InverseGamma(40.0, 10.0)
  val priorW = InverseGamma(40.0, 10.0)

  val initP = DlmParameters(
    v = DenseMatrix(priorV.draw),
    w = diag(DenseVector.fill(2)(priorW.draw)),
    m0 = p.m0,
    c0 = p.c0
  )

  val iters = GibbsSampling
    .sample(mod, priorV, priorW, initP, data)
    .steps
    .drop(12000)
    .take(12000)

  val out = new java.io.File("examples/data/gibbs_spain_investment.csv")
  val writer = out.asCsvWriter[List[Double]](rfc.withHeader("V", "W1", "W2"))

  def formatParameters(p: DlmParameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.p))
  }

  writer.close()
}
