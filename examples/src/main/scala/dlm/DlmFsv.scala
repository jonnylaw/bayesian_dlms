package examples.dlm

import core.dlm.model._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions._
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import akka.actor.ActorSystem
import akka.stream._
import scaladsl._

trait DlmFsvModel {
  val mod = List.fill(6)(Dlm.polynomial(1)).reduce(_ |*| _)

  // simulate data
  val beta = DenseMatrix(
    (1.0,  0.0),
    (0.3,  1.0),
    (0.07, 0.25),
    (0.23, 0.23),
    (0.4,  0.25),
    (0.2,  0.23))

  val params = FactorSv.Parameters(v = 0.1, beta,
    Vector.fill(2)(SvParameters(0.8, 2.0, 0.2))
  )

  val dlmP = DlmParameters(
    v = diag(DenseVector.fill(6)(2.0)),
    w = diag(DenseVector.fill(6)(3.0)),
    DenseVector.fill(6)(0.0),
    diag(DenseVector.fill(6)(1.0)))

  val p = DlmFsv.Parameters(dlmP, params)
}

trait SimulatedDlmFsv {
  val rawData = Paths.get("examples/data/dlm_fsv_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Dlm.Data(a.head, DenseVector(a.slice(1, 7).map(_.some).toArray))
  }.toVector
}

object SimulateDlmFsv extends App with DlmFsvModel {
  val sims = DlmFsv.simulate(mod, p).steps.take(1000)

  val out = new java.io.File("examples/data/dlm_fsv_sims.csv")
  val names: Seq[String] = Seq("time") ++ (1 to 6).map(i => s"observation_$i") ++ (1 to 6).map(i => s"state_$i") ++ (1 to 2).map(i => s"log-variance_$i")
  val headers = rfc.withHeader(names: _*)
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

object ParametersDlmFsv
    extends App
    with DlmFsvModel
    with SimulatedDlmFsv {

  implicit val system = ActorSystem("dlm_fsv")
  implicit val materializer = ActorMaterializer()

  val priorBeta = Gaussian(0.0, 5.0)
  val priorSigmaEta = InverseGamma(1, 0.01)
  val priorPhi = new Beta(5, 2)
  val priorMu = Gaussian(0.0, 3.0)
  val priorSigma = InverseGamma(1, 0.01)
  val priorW = InverseGamma(1, 0.01)

  val iters = DlmFsv.sample(priorBeta, priorSigmaEta, priorPhi,
    priorMu, priorSigma, priorW, data, mod, p)

  def formatParameters(s: DlmFsv.State): List[Double] = {
    s.p.toList
  }

  // write iters to file
  Streaming.writeParallelChain(
    iters, 2, 10000, "examples/data/dlm_fsv_params", formatParameters).
    runWith(Sink.onComplete(_ => system.terminate()))
}
