package dlm.examples

import dlm.model._
import cats.implicits._
import Dlm._
import GibbsSampling._
import breeze.linalg._
import breeze.stats.distributions._
import java.nio.file.Paths
import kantan.csv._
import kantan.csv.ops._
import kantan.csv.generic._

trait CorrelatedModel {
  val mod1 = polynomial(1)
  val mod2 = polynomial(1)

  val model = Dlm.outerSumModel(mod1, mod2)

  val v = diag(DenseVector(1.0, 2.0))
  val w = DenseMatrix((0.75, 0.5), (0.5, 1.25))
  val c0 = diag(DenseVector(100.0, 100.0))

  val p = Parameters(
    v,
    w,
    DenseVector.zeros[Double](2),
    c0
  )
}

object SimulateCorrelated extends App with CorrelatedModel {
  val sims = Dlm.simulate(0, model, p).
    steps.
    take(1000)

  val out = new java.io.File("data/CorrelatedDlm.csv")
  val headers = rfc.withHeader("time", "observation_1", "observation_2", "state_1", "state_2")
  val writer = out.asCsvWriter[(Time, Option[Double], Option[Double], Double, Double)](headers)

  def formatData(d: (Data, DenseVector[Double])) = d match {
    case (Data(t, y), x) =>
      (t, y.map(x => x(0)), y.map(x => x(1)), x(0), x(1))
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

object GibbsCorrelated extends App with CorrelatedModel {
  val rawData = Paths.get("data/CorrelatedDlm.csv")
  val reader = rawData.asCsvReader[(Time, Double, Double, Double, Double)](rfc.withHeader)
  val data = reader.
    collect { 
      case Success(a) => Data(a._1, DenseVector(a._2, a._3).some)
    }.
    toArray

  val iters = gibbsSamples(model, Gamma(1.0, 10.0), Gamma(1.0, 10.0), p, data).
    steps.
    take(10000)

  val out = new java.io.File("data/correlated_dlm_gibbs.csv")
  val headers = rfc.withHeader("V1", "V2", "V3", "V4", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatParameters(p: Parameters) = {
    (p.v.data ++ p.w.data).toList
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.p))
  }

  writer.close()
}
