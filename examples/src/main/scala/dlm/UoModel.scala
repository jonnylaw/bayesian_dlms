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
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

trait JointUoModel {
  val format = DateTimeFormatter.ISO_DATE_TIME
  implicit val codec: CellCodec[LocalDateTime] =
    localDateTimeCodec(format)
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
  } yield FsvParameters(sigma,
    FactorSv.buildBeta(4, 2, bij),
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
  } yield List(foP, seasonalP, foP, seasonalP).
    reduce(Dlm.outerSumParameters)

  val initP: Rand[DlmFsvParameters] = for {
    fsv <- fsvP
    dlm <- dlmP
  } yield DlmFsvParameters(dlm, fsv) 

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
    filter(_.datetime.compareTo(
      LocalDateTime.of(2017, Month.SEPTEMBER, 1, 0, 0)) > 0).
    filter(_.datetime.compareTo(
      LocalDateTime.of(2017, Month.OCTOBER, 1, 0, 0)) < 0).
    // time in hours
    map(a => Data(a.datetime.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0), 
        DenseVector(a.co, a.humidity, a.no, a.temperature))).
    toVector.
    sortBy(_.time).
    zipWithIndex.
    filter { case (_, i) => i % 10 == 0 }. // thinned
    map(_._1)
}

object FitContUo extends App with JointUoModel {
  implicit val system = ActorSystem("dlm_fsv_uo")
  implicit val materializer = ActorMaterializer()

  val iters = DlmFsv.sampleOu(priorBeta, priorSigmaEta,
    priorPhi, priorMu, priorSigma, priorW,
    data, dlmComp, initP.draw)

  def diagonal(m: DenseMatrix[Double]): List[Double] =
    for {
      i <- List.range(0, m.cols)
    } yield m(i,i)

  def format(s: DlmFsv.State): List[Double] =
    s.p.toList

  // write iterations
  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/uo_cont_gibbs_two_factors", format).
    runWith(Sink.onComplete(_ => system.terminate()))
}

object InterpolateUo extends App with JointUoModel {
  def readValue(a: String): Option[Double] = 
    if (a == "NA") None else Some(a.toDouble)

  val file = "examples/data/uo_cont_gibbs_two_factors_0.csv"

  // read in parameters
  val rawMcmc = Paths.get(file)
  val reader2 = rawMcmc.asCsvReader[List[Double]](rfc.withHeader)
  val ps: DlmFsvParameters = reader2.
    collect {
      case Right(a) => a
    }.
    map(x => DlmFsvParameters.fromList(4, 16, 4, 2)(x)).
    foldLeft((DlmFsvParameters.empty(4, 16, 4, 2), 1.0)){
      case ((avg, n), b) =>
        (avg.map(_ * n).add(b).map(_  / (n + 1)), n + 1)
    }._1

  // read in data with encoded missing bits
  val rawData1 = Paths.get("examples/data/new_new_emote_1108_wide.csv")
  val reader1 = rawData1.asCsvReader[(LocalDateTime, String, String, String, String)](rfc.withHeader)
  val testData = reader1.
    collect {
      case Right(a) => Data(a._1.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0),
        DenseVector(readValue(a._2), readValue(a._3),
          readValue(a._4), readValue(a._5)))
    }.
    toVector.
    // zipWithIndex.
    // filter { case (_, i) => i % 12 == 0 }. // thinned
    // map(_._1).
    sortBy(_.time)

  // use mcmc to sample the missing observations
  val iters = DlmFsv.sampleStateOu(testData, dlmComp, ps).
    steps.
    take(1000).
    map { (s: DlmFsv.State) =>
      val vol = s.volatility.map(x => (x.time, x.sample))
      val st = s.theta.map(x => (x.time, x.sample))
      DlmFsv.obsVolatility(vol, st, dlmComp, ps)
    }.
    map(s => s.map { case (t, y) => (t, Some(y(0))) }).
    toVector

  val summary = DlmFsv.summariseInterpolation(iters, 0.995)

  // write interpolated data
  val out = new java.io.File("data/interpolated_urbanobservatory.csv")
  val headers = rfc.withHeader("time", "mean", "upper", "lower")

  out.writeCsv(summary, headers)
}
