package examples.dlm

import core.dlm.model._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions._
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import akka.actor.ActorSystem
import akka.stream._
import scaladsl._

trait FsvModel {
  // simulate data
  val beta = DenseMatrix(
    (1.0,  0.0),
    (0.3,  1.0),
    (0.07, 0.25),
    (0.23, 0.23),
    (0.4,  0.25),
    (0.2,  0.23))

  val params = FactorSv.Parameters(
    v = 0.1,
    beta,
    Vector.fill(2)(SvParameters(0.8, 2.0, 0.2))
  )
}
/**
  * Simulate a stochastic volatility model with an AR(1) latent state
  */
object SimulateFsv extends App with FsvModel {

  val sims = FactorSv.simulate(params).steps.take(1000)

  // write to file
  val out = new java.io.File("examples/data/fsv_sims.csv")
  val colnames = Seq("time") ++ (1 to 6).map(i => s"observation_$i") ++
    (1 to 2).map(i => s"factor_$i") ++ (1 to 2).map(i => s"log-volatility_$i")
  val headers = rfc.withHeader(colnames: _*)
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatData(d: (Dlm.Data, Vector[Double], Vector[Double])): List[Double] = d match {
    case (Dlm.Data(t, y), f, a) =>
      List(t) ::: y.data.toList.flatten ::: f.toList ::: a.toList
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

object FitFsv extends App with FsvModel {
  implicit val system = ActorSystem("factor_sv")
  implicit val materializer = ActorMaterializer()

  val rawData = Paths.get("examples/data/fsv_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Dlm.Data(a.head, DenseVector(a.slice(1, 7).toArray.map(_.some)))
  }.toVector

  val priorBeta = Gaussian(1.0, 5.0)
  val priorSigmaEta = InverseGamma(2.5, 1.0)
  val priorPhi = new Beta(20, 2)
  val priorMu = Gaussian(2.0, 1.0)
  val priorSigma = InverseGamma(2.5, 3.0)

  val iters = FactorSv
    .sampleAr(priorBeta, priorSigmaEta, priorMu, priorPhi, priorSigma, data, params)

  def formatParameters(s: FactorSv.State) = s.params.toList

  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/factor_sv_gibbs", formatParameters).
    runWith(Sink.onComplete(_ => system.terminate()))
}
