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

  case class Radiance(
    stationId:    Int,
    datetime:     LocalDateTime,
    forecastHour: Int,
    radiance:     Double,
    forecast:     Double)

  val rawData = Paths.get("examples/data/forecast_actual_radiance_training.csv")
  val reader = rawData.asCsvReader[Radiance](rfc.withHeader)
  val data: Vector[Radiance] = reader
    .collect {
      case Right(a) => a
    }
    .toVector

  val ys = data.
    filter(d => d.stationId == 8 || d.stationId == 25 || d.stationId == 49 ||
      d.stationId == 57 || d.stationId == 70 || d.stationId == 77).
    groupBy(_.datetime).
    map { case (date, readings) => (Data(date.toEpochSecond(ZoneOffset.UTC),
      DenseVector(readings.map(_.radiance.some).toArray)),
      readings.map(_.forecast))
    }.
    toVector.
    sortBy(_._1.time).
    zipWithIndex.
    map { case (d, t) => (Data(t + 1, d._1.observation), d._2) }

  // forecast should be length 6, with an array of "predictors" for each station
  val forecast = ys.map(_._2).
    transpose.
    map(_.map((x: Double) => DenseVector(x)).toArray)

  // radiance should be a time series with 6 dimensional measurements
  val actual = ys.map(_._1).toVector

  val rawDataTest = Paths.get("examples/data/forecast_actual_radiance_test.csv")
  val readerTest = rawDataTest.asCsvReader[Radiance](rfc.withHeader)
  val dataTest = readerTest.
    collect {
      case Right(a) => a
    }.
    toVector

  val ysTest = dataTest.
    filter(d => d.stationId == 8 || d.stationId == 25 || d.stationId == 49 ||
      d.stationId == 57 || d.stationId == 70 || d.stationId == 77).
    groupBy(_.datetime).
    map { case (date, readings) => (Data(date.toEpochSecond(ZoneOffset.UTC),
      DenseVector(readings.map(_.radiance.some).toArray)),
      readings.map(_.forecast))
    }.
    toVector.
    sortBy(_._1.time).
    zipWithIndex.
    map { case (d, t) => (Data(t + 1, d._1.observation), d._2) }

  // forecast should be length 6, with an array of "predictors" for each station
  val forecastTest = ysTest.map(_._2).
    transpose.
    map(_.map((x: Double) => DenseVector(x)).toArray)
}

object RadianceRegression extends App with ReadRadianceData {
  implicit val system = ActorSystem("radiance_regression")
  implicit val materializer = ActorMaterializer()

  // Build a multivariate regression DLM
  def model(x: Vector[Array[DenseVector[Double]]]) =
    x.map(xi => Dlm.regression(xi)).
      reduce(_ |*| _)

  val dlmP = DlmParameters(
    v = diag(DenseVector.fill(6)(1.0)),
    w = diag(DenseVector.fill(12)(1.0)),
    m0 = DenseVector.fill(12)(0.0),
    c0 = diag(DenseVector.fill(12)(10.0))
  )

  def format(s: GibbsSampling.State): List[Double] =
    s.p.toList

  val priorW = InverseGamma(10.0, 1.0)
  val priorV = InverseGamma(10.0, 1.0)

  val iters = GibbsSampling.sampleSvd(model(forecast.toVector), priorV,
    priorW, dlmP, actual.toVector)

  // write iters to file
  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/radiance_regression_params", format).
    runWith(Sink.onComplete(_ => system.terminate()))
}

object ForecastRadiance extends App with ReadRadianceData {
  // Build a multivariate regression DLM
  def model(x: Vector[Array[DenseVector[Double]]]) =
    x.map(xi => Dlm.regression(xi)).
      reduce(_ |*| _)

  // read in the parameters from the MCMC chain and caculate the mean
  val mcmcChain = Paths.get("examples/data/radiance_regression_params_0.csv")
  val read = mcmcChain.asCsvReader[List[Double]](rfc.withHeader)

  val params: List[Double] =
    read.collect { case Right(a) => a }.toList.transpose.map(a => mean(a))

  val meanParameters = DlmParameters(
    diag(DenseVector(params.take(6).toArray)),
    diag(DenseVector(params.drop(6).take(12).toArray)),
    DenseVector.fill(12)(0.0),
    diag(DenseVector.fill(12)(10.0)))

  // get the posterior distribution of the final state
  val filtered = SvdFilter(SvdFilter.advanceState(meanParameters, model(forecast).g)).
    filter(model(forecast), actual, meanParameters)
  val (mt, ct, initTime) = filtered.map { a =>
    val ct = a.uc * diag(a.dc) * a.uc.t
    (a.mt, ct, a.time)
  }.last

  val forecasted: List[List[Double]] = Dlm.forecast(model(forecastTest),
    mt, ct, 1.0, meanParameters).
    map { case (t, ft, qt) =>
      t :: Dlm.summariseForecast(ft, qt).toList.flatten }.
    take(12).
    toList

  val out = new java.io.File("examples/data/radiance_regression_forecast.csv")
  val headers = rfc.withHeader("time" :: (1 to 6).
    flatMap(i => List(s"forecast_mean_$i", s"lower_$i", s"upper_$i")).toList: _*)
  val writer = out.writeCsv(forecasted, headers)
}

trait RadFactorModel {
 // Build a multivariate regression DLM
  def model(x: Vector[Array[DenseVector[Double]]]) =
    x.map(xi => Dlm.regression(xi)).
      reduce(_ |*| _)

  val dlmP = DlmParameters(
    v = diag(DenseVector.fill(6)(1.0)),
    w = diag(DenseVector.fill(12)(1.0)),
    m0 = DenseVector.fill(12)(0.0),
    c0 = diag(DenseVector.fill(12)(10.0))
  )

  val factorP = FactorSv.Parameters(
    v = 1.0,
    beta = FactorSv.makeBeta(6, 2),
    factorParams = Vector.fill(2)(SvParameters(0.5, 0.0, 0.1)))
  val p = DlmFsv.Parameters(dlmP, factorP)
}

object RadianceFactors extends App with ReadRadianceData with RadFactorModel {
  implicit val system = ActorSystem("radiance_regression")
  implicit val materializer = ActorMaterializer()

  val priorBeta = Gaussian(0.0, 5.0)
  val priorSigma = InverseGamma(3.0, 5.0)
  val priorW = InverseGamma(10.0, 1.0)
  val priorSigmaEta = InverseGamma(3.0, 5.0)
  val priorPhi = new Beta(5, 2)
  val priorMu = Gaussian(0.0, 5.0)

  def diagonal(m: DenseMatrix[Double]) = {
    for {
      i <- List.range(0, m.cols)
    } yield m(i, i)
  }

  def format(s: DlmFsv.State): List[Double] = {
    s.p.fsv.v :: s.p.fsv.beta.data.toList ::: diagonal(s.p.dlm.w) :::
    s.p.fsv.factorParams.toList.flatMap(_.toList)
  }

  val iters = DlmFsv.sample(priorBeta, priorSigmaEta, priorPhi, priorMu,
    priorSigma, priorW, actual, model(forecast), p)

  // write iters to file
  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/radiance_regression_factor_params", format).
    runWith(Sink.onComplete(_ => system.terminate()))
}

object RadianceFactorsSystem extends App with ReadRadianceData with RadFactorModel {
  implicit val system = ActorSystem("radiance_regression_system")
  implicit val materializer = ActorMaterializer()

  val factorParams = FactorSv.Parameters(
    v = 1.0,
    beta = FactorSv.makeBeta(12, 2),
    factorParams = Vector.fill(2)(SvParameters(0.5, 0.0, 0.1)))

  val sp = DlmFsvSystem.Parameters(
    dlmP.m0,
    dlmP.c0,
    v = 1.0,
    factorParams)

  val priorBeta = Gaussian(0.0, 5.0)
  val priorSigma = InverseGamma(3.0, 5.0)
  val priorW = InverseGamma(10.0, 1.0)
  val priorSigmaEta = InverseGamma(3.0, 5.0)
  val priorPhi = new Beta(5, 2)
  val priorMu = Gaussian(0.0, 5.0)

  def diagonal(m: DenseMatrix[Double]) = {
    for {
      i <- List.range(0, m.cols)
    } yield m(i, i)
  }

  def format(s: DlmFsvSystem.State): List[Double] = {
    s.p.v :: s.p.factors.v :: s.p.factors.beta.data.toList :::
    s.p.factors.factorParams.toList.flatMap(_.toList)
  }

  val iters = DlmFsvSystem.sample(priorBeta, priorSigmaEta, priorPhi, priorMu,
    priorSigma, priorW, actual, model(forecast), sp)

  // write iters to file
  Streaming.writeParallelChain(
    iters, 2, 100000, "examples/data/radiance_regression_factor_system_params", format).
    runWith(Sink.onComplete(_ => system.terminate()))
}

object ForecastRadianceFactors extends App with ReadRadianceData {
  // Build a multivariate regression DLM
  def model(x: Vector[Array[DenseVector[Double]]]) =
    x.map(xi => Dlm.regression(xi)).
      reduce(_ |*| _)

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

/**
  * Parameter inference in parallel
  */
object RadianceRegressionDlm extends App with ReadRadianceData {
  implicit val system = ActorSystem("NationalGridRegression")
  implicit val materializer = ActorMaterializer()

  val model = (x: Array[DenseVector[Double]]) =>
    Dlm.regression(x)

  val initP = DlmParameters(
    v = diag(DenseVector.fill(1)(1.0)),
    w = diag(DenseVector.fill(2)(1.0)),
    m0 = DenseVector.fill(2)(0.0),
    c0 = diag(DenseVector.fill(2)(10.0))
  )

  val priorV = InverseGamma(3.0, 5.0)
  val priorW = InverseGamma(3.0, 5.0)

  def mcmc(
    actualRadiance: Vector[Data],
    forecastRadiance: Array[DenseVector[Double]]) =
    GibbsSampling.sample(model(forecastRadiance),
      priorV, priorW, initP, actualRadiance)

  def format(s: GibbsSampling.State): List[Double] = 
    DenseVector.vertcat(diag(s.p.v), diag(s.p.w),
      s.state.last.mean, diag(s.state.last.cov)).data.toList

  Source(data).
    filter(_.stationId == 25).
    groupBy(1000, a => (a.stationId, a.forecastHour)).
    fold((0, 0, Vector.empty[(LocalDateTime, Double, Double)])){(l, r) =>
      (r.stationId, r.forecastHour, l._3 ++ Vector((r.datetime, r.radiance, r.forecast)))
    }.
    mapAsyncUnordered(2){ case (sid, hour, d) => 

      // extract forecast and actual radiance
      val actual = d.map { case (date, r, _) =>
        Data(date.toEpochSecond(ZoneOffset.UTC), DenseVector(Some(r))) }.
        sortBy(_.time).
        zipWithIndex.
        map { case (d, i) => Data(i + 1, d.observation) }

      val forecast = d.map { case (date, _, f) => DenseVector(f) }.toArray

      // run two parallel chains
      Streaming.writeParallelChain(
        mcmc(actual, forecast), 2, 100000,
        s"examples/data/station_${sid}_${hour}_regression", format).
        toMat(Sink.ignore)(Keep.right).
        run()
    }.
    mergeSubstreams.
    runWith(Sink.onComplete { s =>
      println(s)
      system.terminate()
    })
}

// object ForecastRadianceRegressionDlm extends App {
//   implicit val system = ActorSystem("NationalGrid")
//   implicit val materializer = ActorMaterializer()

//   val model = (x: Array[DenseVector[Double]]) => Dlm.regression(x)

//   def readParams(stationId: Int, fcstTime: Int): Future[DlmParameters] =
//     readCsv(s"data/output/station_${stationId}_${fcstTime}_regression.csv").
//       map { ps =>
//         val v = ps.take(1).map(_.toDouble).toArray
//         val w = ps.drop(1).take(2).map(_.toDouble).toArray

//         val p = DlmParameters(diag(DenseVector(v)),
//           diag(DenseVector(w)),
//           DenseVector.zeros[Double](2),
//           DenseMatrix.eye[Double](2))

//         (p, DenseVector(ps.drop(3).map(_.toDouble).toArray))
//       }.
//       runWith(Sink.seq).
//       map { ps =>
//         val (mt, ct) = Dglm.meanCovSamples(ps.map(_._2))
//         meanParameters(ps.map(_._1)).copy(m0 = mt, c0 = ct)
//       }

//   readCsv("data/forecast_actual_radiance_test.csv").
//     map(parseForecastActual).
//     collect { case Some(a) => a }.
//     groupBy(200, a => (a.stationId, a.forecastHour)).
//     fold((0, 0, Vector.empty[(LocalDateTime, Double, Double)])){(l, r) =>
//       (r.stationId, r.forecastHour, l._3 :+ (r.datetime, r.radiance, r.forecastRadiance))
//     }.
//     mapAsyncUnordered(2) { case (stationId, fcstTime, data) =>

//       for {
//         p <- readParams(stationId, fcstTime)
//         fcstRadiance = data.map(a => DenseVector(a._3)).toArray
//         times = data.indices.toVector.map(_.toDouble)
//         f = forecastRadiance(model(fcstRadiance), p, p.m0, p.c0, times)
//         io <- Source.single(f).
//           mapConcat(identity).
//           map { case (t, ft, qt) => (ft, qt) }.
//           zip(Source.single(data).mapConcat(identity)).
//           map { case (f, d) => Vector(d._1, d._3, f._1(0), f._2(0,0)) }.
//           map(_.mkString(",")).
//           map(s => ByteString(s + "\n")).
//           runWith(FileIO.toPath(Paths.get(s"data/output/forecast_station_${stationId}_regression.csv")))
//       } yield io
//     }.
//     mergeSubstreams.
//     runWith(Sink.onComplete { s =>
//       println(s)
//       system.terminate()
//     })
// }

// object LogRadianceRegressionDlm extends App {
//   implicit val system = ActorSystem("NationalGridRegression")
//   implicit val materializer = ActorMaterializer()

//   val model = (x: Array[DenseVector[Double]]) => Dlm.regression(x)

//   val initP = DlmParameters(
//     v = diag(DenseVector.fill(1)(1.0)),
//     w = diag(DenseVector.fill(2)(1.0)),
//     m0 = DenseVector.fill(2)(0.0),
//     c0 = diag(DenseVector.fill(2)(10.0))
//   )

//   val priorV = InverseGamma(3.0, 5.0)
//   val priorW = InverseGamma(3.0, 5.0)

//   def mcmc(actualRadiance: Vector[Dlm.Data], forecastRadiance: Array[DenseVector[Double]]) = 
//     GibbsSampling.sample(model(forecastRadiance), priorV, priorW, initP, actualRadiance).
//       steps.
//       take(10000)

//   def format(s: GibbsSampling.State): List[Double] = {
//     DenseVector.vertcat(diag(s.p.v), diag(s.p.w), s.state.last._2).data.toList
//   }

//   readCsv("data/forecast_actual_radiance_training.csv").
//     map(parseForecastActual).
//     collect { case Some(a) => a }.
//     filter(_.stationId == 25).
//     map(a => a.copy(radiance = math.log(a.radiance + 1),
//       forecastRadiance = math.log(a.forecastRadiance + 1))).
//     groupBy(200, a => (a.stationId, a.forecastHour)).
//     fold((0, 0, Vector.empty[(LocalDateTime, Double, Double)])){(l, r) =>
//       (r.stationId, r.forecastHour, l._3 :+ (r.datetime, r.radiance, r.forecastRadiance))
//     }.
//     mapAsyncUnordered(2){ case (sid, hour, d) => 
//       val actual = d.map { case (date, r, _) =>
//         Dlm.Data(datetimeToSeconds(date), DenseVector(Some(r))) }.
//         sortBy(_.time).
//         zipWithIndex.
//         map { case (d, i) => Dlm.Data(i, d.observation) }

//       val forecast = d map { case (date, _, f) => DenseVector(f) }

//       Source.fromIterator(() => mcmc(actual, forecast.toArray)).
//         runWith(Write.writeChain(format, s"data/output/station_${sid}_${hour}_log_regression.csv"))
//     }.
//     mergeSubstreams.
//     runWith(Sink.onComplete { s =>
//       println(s)
//       system.terminate()
//     })
// }

// object ForecastLogRadianceRegressionDlm extends App {
//   implicit val system = ActorSystem("NationalGrid")
//   implicit val materializer = ActorMaterializer()

//   val model = (x: Array[DenseVector[Double]]) => Dlm.regression(x)

//   def readParams(stationId: Int, fcstTime: Int): Future[DlmParameters] =
//     readCsv(s"data/output/station_${stationId}_${fcstTime}_regression.csv").
//       map { ps =>
//         val v = ps.take(1).map(_.toDouble).toArray
//         val w = ps.drop(1).take(2).map(_.toDouble).toArray

//         val p = DlmParameters(diag(DenseVector(v)),
//           diag(DenseVector(w)),
//           DenseVector.zeros[Double](2),
//           DenseMatrix.eye[Double](2))

//         (p, DenseVector(ps.drop(3).map(_.toDouble).toArray))
//       }.
//       runWith(Sink.seq).
//       map { ps =>
//         val (mt, ct) = Dglm.meanCovSamples(ps.map(_._2))
//         meanParameters(ps.map(_._1)).copy(m0 = mt, c0 = ct)
//       }

//   readCsv("data/forecast_actual_radiance_test.csv").
//     map(parseForecastActual).
//     collect { case Some(a) => a }.
//     filter(_.stationId == 25).
//     map(a => a.copy(radiance = math.log(a.radiance + 1),
//       forecastRadiance = math.log(a.forecastRadiance + 1))).
//     groupBy(200, a => (a.stationId, a.forecastHour)).
//     fold((0, 0, Vector.empty[(LocalDateTime, Double, Double)])){(l, r) =>
//       (r.stationId, r.forecastHour, l._3 :+ (r.datetime, r.radiance, r.forecastRadiance))
//     }.
//     mapAsyncUnordered(2) { case (stationId, fcstTime, data) =>

//       for {
//         p <- readParams(stationId, fcstTime)
//         fcstRadiance = data.map(a => DenseVector(a._3)).toArray
//         times = data.indices.toVector.map(_.toDouble)
//         f = forecastRadiance(model(fcstRadiance), p, p.m0, p.c0, times)
//         io <- Source.single(f).
//           mapConcat(identity).
//           map { case (t, ft, qt) => (ft, qt) }.
//           zip(Source.single(data).mapConcat(identity)).
//           map { case (f, d) => Vector(d._1, d._3, f._1(0), f._2(0,0)) }.
//           map(_.mkString(",")).
//           map(s => ByteString(s + "\n")).
//           runWith(FileIO.toPath(Paths.get(s"data/output/forecast_station_${stationId}_log_regression.csv")))
//       } yield io
//     }.
//     mergeSubstreams.
//     runWith(Sink.onComplete { s =>
//       println(s)
//       system.terminate()
//     })
// }
