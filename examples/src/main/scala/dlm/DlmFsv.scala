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
import plot._
import com.cibo.evilplot.plot.aesthetics.DefaultTheme._

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

  val k = 2 // equivalent to number of columns in beta
  val params = FactorSv.Parameters(v = 0.1, beta,
    Vector.fill(k)(SvParameters(0.8, 2.0, 0.2))
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
  val names: Seq[String] = Seq("time") ++
    (1 to 6).map(i => s"observation_$i") ++
    (1 to 6).map(i => s"state_$i") ++
    (1 to 2).map(i => s"log_variance_$i")

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

object ParametersDlmFsv extends App with DlmFsvModel with SimulatedDlmFsv {
  // implicit val system = ActorSystem("dlm_fsv")
  // implicit val materializer = ActorMaterializer()

  val priorBeta = Gaussian(0.0, 5.0)
  val priorSigmaEta = InverseGamma(10, 1)
  val priorPhi = new Beta(5, 2)
  val priorMu = Gaussian(0.0, 3.0)
  val priorSigma = InverseGamma(10, 1)
  val priorW = InverseGamma(10, 1)

  val iters = DlmFsv.sample(priorBeta, priorSigmaEta, priorPhi,
    priorMu, priorSigma, priorW, data, mod, p)

  def formatParameters(s: DlmFsv.State) = s.p.toList

  iters.
    steps.
    take(100).
    map(formatParameters).
    foreach(println)


  // write iters to file
  // Streaming.writeParallelChain(
  //   iters, 2, 10000, "examples/data/dlm_fsv_params", formatParameters).
  //   runWith(Sink.onComplete(_ => system.terminate()))
}

trait DlmFsvSystemModel {
    val beta = DenseMatrix((1.0, 0.0),
    (4.94, 1.0),
    (3.38, 2.74),
    (5.00, 0.95),
    (-4.57, -7.4),
    (-0.12, 8.2),
    (1.35, -2.8),
    (-10.61, -2.2),
    (-1.39, -0.75),
    (2.15, 0.31),
    (-1.16, 1.46),
    (2.07, 2.18),
    (12.52, -4.0))

  val dlmMod = Dlm.polynomial(1) |+| Dlm.seasonal(24, 3) |+| Dlm.seasonal(24 * 7, 3)

  // use two factors for the matrix
  val errorP = FactorSv.Parameters(
    v = 0.1,
    beta,
    Vector.fill(2)(SvParameters(0.2, 0.0, 0.2))
  )

  val params = DlmFsvSystem.Parameters(
    m0 = DenseVector.zeros[Double](13),
    c0 = DenseMatrix.eye[Double](13),
    0.5, errorP)
}

object SimulateDlmFsvSystem extends App with DlmFsvSystemModel {
  val sims = DlmFsvSystem.simulateRegular(dlmMod, params, 1).
    steps.
    take(5000)

  val out = new java.io.File("examples/data/dlm_fsv_system_sims.csv")
  val names: Seq[String] = Seq("time", "observation") ++
    (1 to 13).map(i => s"state_$i") ++
    (1 to 2).map(i => s"log_variance_$i")

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

object FitDlmFsvSystem extends App with DlmFsvSystemModel {
  // implicit val system = ActorSystem("dlm_fsv_system")
  // implicit val materializer = ActorMaterializer()

  val rawData = Paths.get("examples/data/dlm_fsv_system_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Dlm.Data(a.head, DenseVector(a.slice(1, 2).map(_.some).toArray))
  }.toVector.
    take(1000)

  val priorBeta = Gaussian(0.0, 5.0)
  val priorSigmaEta = InverseGamma(0.01, 0.01)
  val priorPhi = new Beta(5, 2)
  val priorMu = Gaussian(0.0, 3.0)
  val priorSigma = InverseGamma(0.01, 0.01)
  val priorV = InverseGamma(0.01, 0.01)

  val initSv = for {
    phi <- priorPhi
    mu <- priorMu
    sigma <- priorSigmaEta
  } yield Vector.fill(2)(SvParameters(phi, mu, sigma))

  // val initP = DlmFsvSystem.Parameters(
  //   DenseVector.fill(13)(0.0),
  //   diag(DenseVector.fill(13)(10.0)),
  //   priorV.draw,
  //   FactorSv.Parameters(
  //     priorSigma.draw,
  //     FactorSv.drawBeta(13, 2, priorBeta).draw,
  //     initSv.draw
  //   )
  // )

  def formatParameters(s: DlmFsvSystem.State): List[Double] = {
    s.p.toList
  }

  val iters = DlmFsvSystem.sample(priorBeta, priorSigmaEta, priorPhi, priorMu,
    priorSigma, priorV, data, dlmMod, params)

  iters.
    steps.
    take(100).
    map(formatParameters).
    foreach(println)

  // write iters to file
  // Streaming.writeParallelChain(
  //   iters, 2, 100000, "examples/data/dlm_fsv_system_params", formatParameters).
  //   runWith(Sink.onComplete(_ => system.terminate()))
}
