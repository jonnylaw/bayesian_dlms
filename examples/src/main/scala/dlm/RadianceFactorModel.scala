package examples.dlm

import dlm.core.model._
import breeze.linalg.{DenseMatrix, DenseVector, sum, diag}
import breeze.stats.distributions.{MarkovChain, Gaussian, Beta}
import breeze.stats.mean
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import kantan.csv.generic._
import java.time._
import java.time.format._
import java.time.temporal.ChronoUnit
import kantan.csv.java8._
import akka.actor.ActorSystem
import akka.stream._
import akka.util.ByteString
import scaladsl._

trait ReadRadianceData {
  val format = DateTimeFormatter.ISO_DATE_TIME
  implicit val codec: CellCodec[LocalDateTime] = localDateTimeCodec(format)

  case class Radiance(stationId: Int,
                      datetime: LocalDateTime,
                      forecastHour: Int,
                      radiance: Double,
                      forecast: Double)

  val rawData = Paths.get("examples/data/forecast_actual_radiance_training.csv")
  val reader = rawData.asCsvReader[Radiance](rfc.withHeader)
  val data: Vector[Radiance] = reader.collect {
    case Right(a) => a
  }.toVector

  val ys = data
    .filter(d =>
      d.stationId == 8 || d.stationId == 25 || d.stationId == 49 ||
        d.stationId == 57 || d.stationId == 70 || d.stationId == 77)
    .groupBy(_.datetime)
    .map {
      case (date, readings) =>
        (Data(date.toEpochSecond(ZoneOffset.UTC),
              DenseVector(readings.map(_.radiance.some).toArray)),
         readings.map(_.forecast))
    }
    .toVector
    .sortBy(_._1.time)
    .zipWithIndex
    .map { case (d, t) => (Data(t + 1, d._1.observation), d._2) }

  // forecast should be length 6, with an array of "predictors" for each station
  val forecast =
    ys.map(_._2).transpose.map(_.map((x: Double) => DenseVector(x)).toArray)

  // radiance should be a time series with 6 dimensional measurements
  val actual = ys.map(_._1).toVector

  val rawDataTest = Paths.get("examples/data/forecast_actual_radiance_test.csv")
  val readerTest = rawDataTest.asCsvReader[Radiance](rfc.withHeader)
  val dataTest = readerTest.collect {
    case Right(a) => a
  }.toVector

  val ysTest = dataTest
    .filter(d =>
      d.stationId == 8 || d.stationId == 25 || d.stationId == 49 ||
        d.stationId == 57 || d.stationId == 70 || d.stationId == 77)
    .groupBy(_.datetime)
    .map {
      case (date, readings) =>
        (Data(date.toEpochSecond(ZoneOffset.UTC),
              DenseVector(readings.map(_.radiance.some).toArray)),
         readings.map(_.forecast))
    }
    .toVector
    .sortBy(_._1.time)
    .zipWithIndex
    .map { case (d, t) => (Data(t + 1, d._1.observation), d._2) }

  // forecast should be length 6, with an array of "predictors" for each station
  val forecastTest =
    ysTest.map(_._2).transpose.map(_.map((x: Double) => DenseVector(x)).toArray)
}

trait RadFactorModel {
  // Build a multivariate regression DLM
  def model(x: Vector[Array[DenseVector[Double]]]) =
    x.map(xi => Dlm.regression(xi)).reduce(_ |*| _)

  val dlmP = DlmParameters(
    v = diag(DenseVector.fill(6)(1.0)),
    w = diag(DenseVector.fill(12)(1.0)),
    m0 = DenseVector.fill(12)(0.0),
    c0 = diag(DenseVector.fill(12)(10.0))
  )

  val factorP = FsvParameters(v = DenseMatrix.eye[Double](6),
                              beta = FactorSv.makeBeta(6, 2),
                              factorParams =
                                Vector.fill(2)(SvParameters(0.5, 0.0, 0.1)))
  val p = DlmFsvParameters(dlmP, factorP)
}

object RadianceFactors extends App with ReadRadianceData with RadFactorModel {
  implicit val system = ActorSystem("radiance_regression")
  implicit val materializer = ActorMaterializer()

  val priorBeta = Gaussian(0.0, 5.0)
  val priorSigma = InverseGamma(3.0, 5.0)
  val priorW = InverseGamma(10.0, 1.0)
  val priorSigmaEta = InverseGamma(3.0, 5.0)
  val priorPhi = Gaussian(0.8, 0.1)
  val priorMu = Gaussian(0.0, 5.0)

  def diagonal(m: DenseMatrix[Double]) = {
    for {
      i <- List.range(0, m.cols)
    } yield m(i, i)
  }

  def format(s: DlmFsv.State): List[Double] = {
    diagonal(s.p.fsv.v) ::: s.p.fsv.beta.data.toList ::: diagonal(s.p.dlm.w) :::
      s.p.fsv.factorParams.toList.flatMap(_.toList)
  }

  val iters = DlmFsv.sample(priorBeta,
                            priorSigmaEta,
                            priorPhi,
                            priorMu,
                            priorSigma,
                            priorW,
                            actual,
                            model(forecast),
                            p)

  // write iters to file
  Streaming
    .writeParallelChain(iters,
                        2,
                        100000,
                        "examples/data/radiance_regression_factor_params",
                        format)
    .runWith(Sink.onComplete(_ => system.terminate()))
}

object RadianceFactorsSystem
    extends App
    with ReadRadianceData
    with RadFactorModel {
  implicit val system = ActorSystem("radiance_regression_system")
  implicit val materializer = ActorMaterializer()

  val factorParams = FsvParameters(
    v = DenseMatrix.eye[Double](6),
    beta = FactorSv.makeBeta(12, 2),
    factorParams = Vector.fill(2)(SvParameters(0.5, 0.0, 0.1)))

  val sp = DlmFsvParameters(dlmP, factorParams)

  val priorBeta = Gaussian(0.0, 5.0)
  val priorSigma = InverseGamma(3.0, 5.0)
  val priorW = InverseGamma(10.0, 1.0)
  val priorSigmaEta = InverseGamma(3.0, 5.0)
  val priorPhi = Gaussian(0.8, 0.1)
  val priorMu = Gaussian(0.0, 5.0)

  def diagonal(m: DenseMatrix[Double]) = {
    for {
      i <- List.range(0, m.cols)
    } yield m(i, i)
  }

  def format(s: DlmFsvSystem.State): List[Double] =
    s.p.toList

  val iters = DlmFsvSystem.sample(priorBeta,
                                  priorSigmaEta,
                                  priorPhi,
                                  priorMu,
                                  priorSigma,
                                  priorW,
                                  actual,
                                  model(forecast),
                                  sp)

  // write iters to file
  Streaming
    .writeParallelChain(
      iters,
      2,
      100000,
      "examples/data/radiance_regression_factor_system_params",
      format)
    .runWith(Sink.onComplete(_ => system.terminate()))
}

object ForecastRadianceFactors extends App with ReadRadianceData {
  // Build a multivariate regression DLM
  def model(x: Vector[Array[DenseVector[Double]]]) =
    x.map(xi => Dlm.regression(xi)).reduce(_ |*| _)

  // read in the parameters from the MCMC chain and caculate the mean
  val mcmcChain = Paths.get("examples/data/radiance_regression_params_0.csv")
  val read = mcmcChain.asCsvReader[List[Double]](rfc.withHeader)

  // TODO: Parse parameters
  // val params: Vector[DlmFsvSystem.Parameters] = read.
  //   collect { case Right(a) => DlmFsvParameters(a,  }.
  //   toVector

  // val k = 100 // perform k samples from the parameter posterior for the forecast
  // val indices = Vector.fill(k)(StochasticVolatilityKnots.discreteUniform(0, params.size))
  // val ps = indices map (i => params(i))

  // // get the posterior distribution of the final state
  // // using the MCMC samples
  // val filtered = SvdFilter(SvdFilter.advanceState(meanParameters, model(forecast).g)).
  //   filter(model(forecast), actual, meanParameters)
  // val (mt, ct, initTime) = filtered.map { a =>
  //   val ct = a.uc * diag(a.dc) * a.uc.t
  //   (a.mt, ct, a.time)
  // }.last

  // val forecasted = DlmFsvSystem.forecast(model(forecastTest), ps, 0.0, 12, 6)

  // val out = new java.io.File("examples/data/radiance_regression_forecast.csv")
  // val headers = rfc.withHeader("time", "forecast", "variance")
  // val writer = out.writeCsv(forecasted, headers)
}
