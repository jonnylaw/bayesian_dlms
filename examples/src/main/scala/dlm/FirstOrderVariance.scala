package examples.dlm

import core.dlm.model._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions._
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._

trait FirstOrderDlmVariance {
  val mod = Dlm.polynomial(1)

  val dlmP = DlmParameters(v = DenseMatrix(2.0),
                           w = DenseMatrix(3.0),
                           DenseVector(0.0),
                           DenseMatrix(1.0))

  val p = DlmSv.Parameters(
    dlmP,
    Vector(SvParameters(0.8, 0.0, 0.2)))
}

trait SimulatedFoDlmVar {
  val rawData = Paths.get("examples/data/first_order_dlm_var.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Dlm.Data(a.head, DenseVector(a(1).some))
  }.toVector
}

object SimulateVarianceDlm extends App with FirstOrderDlmVariance {
  val sims = DlmSv.simulate(mod, p, 1).steps.take(1000)

  val out = new java.io.File("examples/data/first_order_dlm_var.csv")
  val headers = rfc.withHeader("time", "observation", "state", "log-variance")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatData(d: (Dlm.Data, DenseVector[Double], Vector[Double])) = d match {
    case (Dlm.Data(t, y), x, a) =>
      t :: KalmanFilter.flattenObs(y).data.toList ::: x.data.toList ::: a.toList
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

object ParametersFirstOrderVariance
    extends App
    with FirstOrderDlmVariance
    with SimulatedFoDlmVar {

  val priorSigma = InverseGamma(1, 0.01)
  val priorPhi = new Beta(5, 2)
  val priorW = InverseGamma(1, 0.01)

  val iters = DlmSv
    .sample(priorW, priorPhi, priorSigma, data, mod, p,
      FilterAr.advanceState, FilterAr.backStep)
    .steps
    .take(100000)
    .map(_.params)
    .map(formatParameters)

  // write iters to file
  val headers = rfc.withHeader("W", "phi", "sigmaEta")
  val out = new java.io.File("examples/data/first_order_dlm_var_params.csv")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatParameters(p: DlmSv.Parameters): List[Double] = {
    p.dlm.w.data(0) :: p.sv.head.phi :: p.sv.head.sigmaEta :: Nil
  }

  while (iters.hasNext) {
    writer.write(iters.next)
  }

  writer.close()
}
