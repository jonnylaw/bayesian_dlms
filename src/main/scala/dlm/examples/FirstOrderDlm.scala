package dlm.examples

import dlm.model._
import Dlm._
import MetropolisHastings._
import GibbsSampling._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{Gamma, Gaussian, Rand}
import java.io.{File, PrintWriter}
import math.{log, exp}
import cats.implicits._

trait FirstOrderDlm {
  val mod = Model(
    f = (t: Time) => DenseMatrix((1.0)), 
    g = (t: Time) => DenseMatrix((1.0))
  )
  val p = Parameters(Vector(3.0), Vector(1.0), Vector(0.0), Vector(1.0))
}

object SimulateDlm extends App with FirstOrderDlm {
  val data = simulate(0, mod, p).
    steps.
    take(1000).
    toArray

  val pw = new PrintWriter(new File("data/FirstOrderDlm.csv"))
  val strings = data.
    map { case (d, x) => s"${d.time}, ${d.observation.get.data.mkString(", ")}, ${x.data.mkString(", ")}" }

  pw.write(strings.mkString("\n"))
  pw.close()    
}

object FilterDlm extends App with FirstOrderDlm {
  val data = scala.io.Source.fromFile("data/FirstOrderDlm.csv").
    getLines.
    map(_.split(",")).
    map(x => Data(x(0).toInt, Some(DenseVector(x(1).toDouble)))).
    toArray

  val filtered = KalmanFilter.kalmanFilter(mod, data, p)

  val pw = new PrintWriter(new File("data/FirstOrderDlmFiltered.csv"))
  pw.write(filtered.mkString("\n"))
  pw.close()
}

object SmoothDlm extends App with FirstOrderDlm {
  val data = scala.io.Source.fromFile("data/FirstOrderDlm.csv").
    getLines.
    map(_.split(",")).
    map(x => Data(x(0).toInt, Some(DenseVector(x(1).toDouble)))).
    toArray

  val filtered = KalmanFilter.kalmanFilter(mod, data, p)
  val smoothed = Smoothing.backwardsSmoother(mod, p)(filtered)

  val pw = new PrintWriter(new File("data/FirstOrderDlmSmoothed.csv"))
  pw.write(smoothed.mkString("\n"))
  pw.close()
}

object LearnParameters extends App with FirstOrderDlm {
  val data: Array[Data] = scala.io.Source.fromFile("data/FirstOrderDlm.csv").
    getLines.
    map(_.split(",")).
    map(x => Data(x(0).toInt, Some(DenseVector(x(1).toDouble)))).
    toArray

  val iters = metropolisHastingsDlm(mod, data, 
    symmetricProposal(0.25), 
    MhState(p, -1e99, 0)).
    steps.
    take(10000)

  // write iters to file
  val pw = new PrintWriter(new File("data/FirstOrderDlmIters.csv"))
  while (iters.hasNext) {
    pw.write(iters.next.toString + "\n")
  }
  pw.close()
}

object GibbsParameters extends App with FirstOrderDlm {
  val data: Array[Data] = scala.io.Source.fromFile("data/FirstOrderDlm.csv").
    getLines.
    map(_.split(",")).
    map(x => Data(x(0).toInt, Some(DenseVector(x(1).toDouble)))).
    toArray

  val iters = gibbsSamples(mod, Gamma(1.0, 10.0), Gamma(1.0, 10.0), p, data).
    steps.
    take(10000)

  // write iters to file
  val pw = new PrintWriter(new File("data/FirstOrderDlmGibbs.csv"))
  // val pw1 = new PrintWriter(new File("data/FirstOrderDlmStateGibbs.csv"))
 while (iters.hasNext) {
    pw.write(iters.next.toString + "\n")
    //    pw1.write(iters.next.state.map(_._2.data).transpose.head.mkString(", ") + "\n")
  }
  // pw.close()
  // pw1.close()
}
