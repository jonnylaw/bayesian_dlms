package dlm.examples

import dlm.model._
import Dlm._
import MetropolisHastings._
import GibbsSampling._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.{Gamma, Gaussian, Rand}
import java.io.{File, PrintWriter}
import math.{log, exp}
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import kantan.csv.generic._

trait DlmModel {
  val mod = polynomial(2)

  val p = Parameters(
    DenseMatrix(3.0),
    diag(DenseVector(2.0, 1.0)),
    DenseVector(0.0, 0.0), 
    diag(DenseVector(100.0, 100.0))
  )
}

object SimulateSecondOrderDlm extends App with DlmModel {
  val data = simulate(0, mod, p).
    steps.
    take(300).
    toArray

  val pw = new PrintWriter(new File("data/SecondOrderDlm.csv"))
  val strings = data.
    map { case (d, x) => s"${d.toString}, ${x.data.mkString(", ")}" }

  pw.write(strings.mkString("\n"))
  pw.close()
  
}

object FilterSecondOrderDlm extends App with DlmModel {
  val data = scala.io.Source.fromFile("data/SecondOrderDlm.csv").
    getLines.
    map(_.split(",")).
    map(x => Data(x(0).toInt, Some(DenseVector(x(1).toDouble)))).
    toArray

  val filtered = KalmanFilter.kalmanFilter(mod, data, p)

  val pw = new PrintWriter(new File("data/SecondOrderDlmFiltered.csv"))
  pw.write(filtered.mkString("\n"))
  pw.close()
}

object SmoothSecondOrderDlm extends App with DlmModel {
  val data = scala.io.Source.fromFile("data/SecondOrderDlm.csv").
    getLines.
    map(_.split(",")).
    map(x => Data(x(0).toInt, Some(DenseVector(x(1).toDouble)))).
    toArray

  val filtered = KalmanFilter.kalmanFilter(mod, data, p)
  val smoothed = Smoothing.backwardsSmoother(mod, p)(filtered)

  val pw = new PrintWriter(new File("data/SecondOrderDlmSmoothed.csv"))
  pw.write(smoothed.mkString("\n"))
  pw.close()
}

object SecondOrderGibbsParameters extends App with DlmModel {
  val data: Array[Data] = scala.io.Source.fromFile("data/SecondOrderDlm.csv").
    getLines.
    map(_.split(",")).
    map(x => Data(x(0).toInt, Some(DenseVector(x(1).toDouble)))).
    toArray

  val alphaV = 1.0/10.0
  val betaV = 1.0/10.0

  val alphaW = 100.0/1000.0
  val betaW = 10.0/1000.0

  val priorV = Gamma(alphaV, 1.0/betaV)
  val priorW = Gamma(alphaW, 1.0/betaW)

  val iters = gibbsSamples(mod, priorV, priorW, p, data).
    steps.
    take(24000)

  // write iters to file
  val pw = new PrintWriter(new File("data/SecondOrderDlmGibbs.csv"))
  while (iters.hasNext) {
    pw.write(iters.next.toString + "\n")
  }
  pw.close()
}

object GibbsInvestParameters extends App with DlmModel {
  val data: Array[Data] = scala.io.Source.fromFile("data/invest2.dat").
    getLines.
    map(_.split(",")).
    zipWithIndex.
    map { case (x, i) => Data(i + 1960, Some(DenseVector(x(1).toDouble))) }.
    toArray

  val alphaV = 1.0/10.0
  val betaV = 1.0/10.0

  val alphaW = 100.0/1000.0
  val betaW = 10.0/1000.0

  val priorV = Gamma(alphaV, 1.0/betaV)
  val priorW = Gamma(alphaW, 1.0/betaW)

  val initP = Parameters(
    v = DenseMatrix(1.0 / priorV.draw),
    w = diag(DenseVector(1.0 / priorW.draw, 1.0 / priorW.draw)),
    m0 = p.m0,
    c0 = p.c0
  )

  val iters = gibbsSamples(mod, priorV, priorW, initP, data).
    steps.
    take(24000)

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
