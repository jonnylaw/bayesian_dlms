package examples.dlm

import dlm.core.model._
import breeze.linalg.{DenseVector, DenseMatrix, diag}
import breeze.stats.distributions._
import kantan.csv._
import kantan.csv.ops._
import kantan.csv.generic._
import java.time._
import java.time.format._
import kantan.csv.java8._
import java.nio.file.Paths
import akka.actor.ActorSystem
import akka.stream._
import scaladsl._

trait AqmeshModel {
  val format = DateTimeFormatter.ISO_DATE_TIME
  implicit val codec: CellCodec[LocalDateTime] =
    localDateTimeCodec(format)

  val seasonalDlm = Dlm.polynomial(1) |+| Dlm.seasonal(12, 3)
  val dlmComp = List.fill(3)(seasonalDlm).reduce(_ |*| _)

  def millisToHours(dateTimeMillis: Long) = 
    dateTimeMillis / 1e3 * 60 * 60

  val priorBeta = Gaussian(0.0, 3.0)
  val priorSigmaEta = InverseGamma(2.5, 3.0)
  val priorPhi = new Beta(20, 2)
  val priorSigma = InverseGamma(2.5, 3.0)
  val priorW = InverseGamma(2.5, 3.0)
  val priorMu = Gaussian(0.0, 1.0)

  val volP = for {
    se <- priorSigmaEta
    phi <- priorPhi
    mu <- priorMu
  } yield SvParameters(phi, mu, se)

  // two factors for the 21 dimensional state
  val fsv = for {
    bij <- priorBeta
    sigmaX <- priorSigma
    vp <- volP
  } yield FsvParameters(
    sigmaX,
    FactorSv.buildBeta(21, 2, bij),
    Vector.fill(2)(vp))

  // the DLM has a 21 dimensional latent state (represented by 3 * 7 models)
  val initP = for {
    fs <- fsv
    v <- InverseGamma(3.0, 0.5)
    m0 = DenseVector.rand(21, Gaussian(0.0, 1.0))
    c0 = DenseMatrix.eye[Double](21)
    dlm = DlmParameters(
      diag(DenseVector.fill(3)(v)),
      DenseMatrix.eye[Double](21), m0, c0),
  } yield DlmFsvParameters(dlm, fs)

  case class AqmeshSensor(
    datetime:      LocalDateTime,
    humidity:      Option[Double],
    no:            Option[Double],
    no2:           Option[Double],
    o3:            Option[Double],
    particleCount: Option[Double],
    pm1:           Option[Double],
    pm10:          Option[Double],
    pm25:          Option[Double],
    pressure:      Option[Double],
    temperature:   Option[Double])

  val rawData = Paths.get("examples/data/aq_mesh_wide.csv")
  val reader = rawData.asCsvReader[AqmeshSensor](rfc.withHeader)
}

object FitAqMesh extends App with AqmeshModel {
  implicit val system = ActorSystem("aqmesh")
  implicit val materializer = ActorMaterializer()

  val training = reader.
    collect { case Right(a) => a }.
    filter(_.datetime.compareTo(
      LocalDateTime.of(2018, Month.JANUARY, 1, 0, 0)) > 0).
    filter(_.datetime.compareTo(
      LocalDateTime.of(2018, Month.FEBRUARY, 1, 0, 0)) < 0).
    toVector.
    zipWithIndex.
    filter { case (_, i) => i % 5 == 0 }. // thinned
    map(_._1).
    map(a => Data(
      a.datetime.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0),
      DenseVector(a.no, a.no2, a.o3))
    ).
    sortBy(_.time)

  val initialParams = initP.draw

  val iters = DlmFsvSystem.sample(priorBeta, priorSigmaEta, priorPhi, priorMu,
    priorSigma, priorW, training, dlmComp, initialParams)

  def formatParameters(s: DlmFsvSystem.State) =
    s.p.toList

  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/aqmesh_gibbs_no_no2_o3", formatParameters).
    runWith(Sink.onComplete { s =>
      println(s)
      system.terminate()
    })
}
