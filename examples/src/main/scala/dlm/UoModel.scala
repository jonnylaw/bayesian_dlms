package examples.dlm

import dlm.core.model._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions._
import kantan.csv._
import kantan.csv.ops._
import kantan.csv.generic._
import java.time._
import java.time.format._
import java.time.temporal.ChronoUnit
import kantan.csv.java8._
import java.nio.file.Paths
import akka.actor.ActorSystem
import akka.stream._
import scaladsl._

trait JointUoModel {
  val format = DateTimeFormatter.ISO_DATE_TIME
  implicit val codec: CellCodec[LocalDateTime] = localDateTimeCodec(format)
  
  val tempDlm = Dlm.polynomial(1) |+| Dlm.seasonal(24, 3)
  val foDlm = Dlm.polynomial(1)
  val dlmComp = List(foDlm, tempDlm, foDlm, tempDlm).reduce(_ |*| _)

  def millisToHours(dateTimeMillis: Long) = {
    dateTimeMillis / 1e3 * 60 * 60
  }

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

  val fsvP = for {
    bij <- priorBeta
    sigma <- priorSigma
    vp <- volP
  } yield FactorSv.Parameters(sigma,
    FactorSv.buildBeta(4, 1, bij),
    Vector.fill(2)(vp))

  val dlmP = for {
    w <- priorW
    foP = DlmParameters(
      v = DenseMatrix.eye[Double](1),
      w = DenseMatrix(w),
      m0 = DenseVector.zeros[Double](1),
      c0 = DenseMatrix.eye[Double](1))
    seasonalP = DlmParameters(
      v = DenseMatrix.eye[Double](1),
      w = diag(DenseVector.fill(7)(w)),
      m0 = DenseVector.zeros[Double](7),
      c0 = DenseMatrix.eye[Double](7))
  } yield List(foP, seasonalP, foP, seasonalP).reduce(Dlm.outerSumParameters)

  val initP: Rand[DlmFsv.Parameters] = for {
    fsv <- fsvP
    dlm <- dlmP
  } yield DlmFsv.Parameters(dlm, fsv) 

  case class EnvSensor(
    datetime:    LocalDateTime,
    co:          Option[Double],
    humidity:    Option[Double],
    no:          Option[Double],
    temperature: Option[Double])

  val rawData = Paths.get("examples/data/new_new_emote_1108_wide.csv")
  val reader = rawData.asCsvReader[EnvSensor](rfc.withHeader)
  val data = reader.
    collect {
      case Right(a) => a
    }.
    // does this work?
    filter(_.datetime.compareTo(LocalDateTime.of(2017, Month.SEPTEMBER, 1, 0, 0)) > 0). 
    filter(_.datetime.compareTo(LocalDateTime.of(2017, Month.OCTOBER, 1, 0, 0)) < 0).
    toVector.
    zipWithIndex.
    filter { case (_, i) => i % 10 == 0 }. // thinned
    map(_._1).
    // time in hours
    map(a => Data(a.datetime.toEpochSecond(ZoneOffset.UTC) / (60 * 60), 
        DenseVector(a.co, a.humidity, a.no, a.temperature))).
    sortBy(_.time)
}

// object FitContUo extends App with JointUoModel {
//   implicit val system = ActorSystem("dlm_fsv_uo")
//   implicit val materializer = ActorMaterializer()

//   val iters = DlmFsv.sampleOu(priorBeta, priorSigmaEta, priorPhi, priorMu,
//     priorSigma, priorW, data, dlmComp, initP.draw)

//   def diagonal(m: DenseMatrix[Double]): List[Double] =
//     for {
//       i <- List.range(0, m.cols)
//     } yield m(i,i)

//   def format(s: DlmFsv.State): List[Double] = {
//     s.p.fsv.v :: s.p.fsv.beta.data.toList ::: diagonal(s.p.dlm.w) :::
//     s.p.fsv.factorParams.toList.flatMap(_.toList)
//   }

//   // write iterations
//   Streaming.writeParallelChain(
//     iters, 2, 100000, "data/uo_cont_gibbs_one_factor", format).
//     runWith(Sink.onComplete(_ => system.terminate()))
// }
