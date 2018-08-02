package examples.dlm

import core.dlm.model._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.Poisson
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import akka.actor.ActorSystem
import akka.stream._
import scaladsl._

trait StudenttDglm {
  val dlm = Dlm.polynomial(1)
  val mod = Dglm.studentT(3, dlm)
  val params = DlmParameters(DenseMatrix(3.0),
                             DenseMatrix(0.1),
                             DenseVector(0.0),
                             DenseMatrix(1.0))
}

trait StudenttData {
  val rawData = Paths.get("examples/data/student_t_dglm.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Dlm.Data(a.head, DenseVector(Some(a(1))))
  }.toVector
}

object SimulateStudentT extends App with StudenttDglm {
  val sims = Dglm.simulateRegular(mod, params, 1.0).steps.take(1000)

  val out = new java.io.File("examples/data/student_t_dglm.csv")
  val header = rfc.withHeader("time", "observation", "state")
  val writer = out.asCsvWriter[List[Double]](header)

  def formatData(d: (Dlm.Data, DenseVector[Double])) = d match {
    case (Dlm.Data(t, y), x) =>
      t :: KalmanFilter.flattenObs(y).data.toList ::: x.data.toList
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

/**
  * Use Kalman Filtering to determine the parameters of the
  * Student's t-distribution DGLM
  */
object StudentTGibbs extends App with StudenttDglm with StudenttData {
  implicit val system = ActorSystem("student-t-gibbs")
  implicit val materializer = ActorMaterializer()

  val priorW = InverseGamma(21.0, 2.0)
  val priorNu = Poisson(3)
  val propNu = (nu: Int) => NegativeBinomial(nu, 0.01)

  val iters =
    StudentT.sample(data.toVector, priorW, priorNu, propNu, mod, params)

  def format(s: StudentT.State): List[Double] = {
    s.nu.toDouble :: DenseVector.vertcat(diag(s.p.v), diag(s.p.w)).data.toList
  }

  Streaming
    .writeParallelChain(iters, 2, 10000, "examples/data/student_t_dglm_gibbs", format)
    .runWith(Sink.onComplete(_ => system.terminate()))
}

object StudentTpmmh extends App with StudenttDglm with StudenttData {
  def prior(p: DlmParameters) = {
    InverseGamma(21.0, 2.0).logPdf(p.w(0, 0)) +
      InverseGamma(5.0, 4.0).logPdf(p.v(0, 0))
  }

  val n = 500
  val iters = Metropolis
    .dglm(mod, data, Metropolis.symmetricProposal(0.01), prior, params, n)
    .steps
    .take(100000)
    .map(_.parameters)

  def formatParameters(p: DlmParameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  val headers = rfc.withHeader("scale", "W")
  Streaming.writeChain(formatParameters,
                       "examples/data/student_t_dglm_pmmh.csv",
                       headers)(iters)
}
