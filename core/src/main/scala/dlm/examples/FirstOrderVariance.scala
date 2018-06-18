package core.dlm.examples

import core.dlm.model._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions._
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._

trait FirstOrderDlmVariance {
  val mod = Dlm.polynomial(1)

  val dlmP = Dlm.Parameters(
    v = DenseMatrix(2.0),
    w = DenseMatrix(3.0), 
    DenseVector(0.0), 
    DenseMatrix(1.0))

  val p = StochasticVolatility.Parameters(dlmP, Vector(0.8), Vector(0.2))
}

trait SimulatedFoDlmVar {
  val rawData = Paths.get("core/data/first_order_dlm_var.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.
    collect { 
      case Right(a) => Dlm.Data(a.head, DenseVector(a(1).some))
    }.
    toVector
}

object SimulateVarianceDlm extends App with FirstOrderDlmVariance {
  val sims = StochasticVolatility.simulate(mod, p, 1).
    steps.
    take(300)

  val out = new java.io.File("core/data/first_order_dlm_var.csv")
  val headers = rfc.withHeader("time", "observation", "state", "log-variance")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatData(d: (Dlm.Data, DenseVector[Double], DenseVector[Double])) = d match {
    case (Dlm.Data(t, y), x, a) =>
      t :: KalmanFilter.flattenObs(y).data.toList ::: x.data.toList ::: a.data.toList
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

object ParametersFirstOrderVariance
    extends App with FirstOrderDlmVariance with SimulatedFoDlmVar {

  val priorSigma = InverseGamma(10, 1)
  val priorPhi = new Beta(5, 2)
  val priorW = InverseGamma(3, 3)

  val iters = StochasticVolatility.sample(priorSigma, priorPhi, priorW, p, data, mod).
    steps.take(10000).map(_.p)

  // write iters to file
  val headers = rfc.withHeader("W", "phi", "sigma")
  val out = new java.io.File("core/data/first_order_dlm_var_params.csv")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatParameters(p: StochasticVolatility.Parameters): List[Double] = {
    p.dlm.w.data(0) :: p.phi.head :: p.sigma.head :: Nil
  }

  while (iters.hasNext) {
    writer.write(formatParameters(iters.next))
  }

  writer.close()
}
