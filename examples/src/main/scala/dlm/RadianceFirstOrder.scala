package examples.dlm

import breeze.linalg.{DenseVector, DenseMatrix, diag}
import dlm.core.model._
import akka.actor.ActorSystem
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import kantan.csv.generic._
import java.time._
import java.time.format._
import kantan.csv.java8._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import akka.stream._
import akka.stream.scaladsl._
import akka.util.ByteString
import akka.NotUsed
import scala.concurrent.Future
import java.nio.file.Paths
import java.time._
import java.time.format._
import java.time.temporal.ChronoUnit

trait ReadRadianceDifference {

  /**
    * An ordering for local Date Time
    */
  implicit def orderDateTime: Ordering[LocalDateTime] =
    Ordering.by(_.toEpochSecond(ZoneOffset.UTC))

  case class Radiance(
      stationId: Int,
      forecastDatetime: LocalDateTime,
      forecastHour: Int,
      difference: Vector[Option[Double]]
  )

  implicit class StringImprovements(val s: String) {
    import scala.util.control.Exception._
    def toIntOpt =
      catching(classOf[NumberFormatException]) opt s.toInt

    def toDatetimeOpt(format: DateTimeFormatter) =
      catching(classOf[DateTimeException]) opt LocalDateTime.parse(s, format)

    def toDoubleOpt =
      catching(classOf[NumberFormatException]) opt s.toDouble
  }

  def writeMeanParams(
      formatParameters: DlmParameters => List[Double],
      file: String): Sink[(Int, Int, DlmParameters), Future[IOResult]] = {

    Flow[(Int, Int, DlmParameters)]
      .map {
        case (id, fsctTime, p) =>
          (id :: fsctTime :: formatParameters(p)).mkString(", ")
      }
      .map(s => ByteString(s + "\n"))
      .toMat(FileIO.toPath(Paths.get(file)))(Keep.right)
  }

  /**
    * Parse a String to a radiance
    */
  def parseRadiance(l: Vector[String]): Option[Radiance] = {
    val format = DateTimeFormatter.ISO_DATE_TIME
    for {
      fcDatetime <- l.head.toDatetimeOpt(format)
      sId <- l(1).toIntOpt
      hour <- l(2).toIntOpt
      // all NAs are actually 0 radiance hence no correction
      diff = l.drop(3).map(_.toDoubleOpt.getOrElse(0.0).some)
    } yield Radiance(sId, fcDatetime, hour, diff)
  }

  def readRadiance(file: String) = {
    FileIO
      .fromPath(Paths.get(file))
      .via(
        Framing.delimiter(ByteString("\n"),
                          maximumFrameLength = 8192,
                          allowTruncation = true))
      .map(_.utf8String)
      .map(a => a.split(",").toVector)
      .map(parseRadiance)
  }

  /**
    * Convert a single radiance reading into a forecast datetime
    * lead time and difference
    */
  def radianceToData(r: Radiance): (Int, Int, Data) =
    (r.stationId,
     r.forecastHour,
     Data(r.forecastDatetime.toEpochSecond(ZoneOffset.UTC) / (60.0 * 60.0),
          DenseVector(r.difference.toArray)))

  /**
    * Given a single station ID, summarise the difference between
    * forecast and actual for each lead time
    * @return a map from lead time and forecast datetime
    * to a vector of data containing the forecast datetime and difference
    * at each lead time
    */
  def radianceToData: Flow[Radiance, (Int, Int, Vector[Data]), NotUsed] = {
    Flow[Radiance]
      .groupBy(100, a => a.stationId)
      .fold((0, 0, Vector.empty[Data])) { (l, r) =>
        val (lt, forecastTime, d) = radianceToData(r)
        (lt, forecastTime, l._3 :+ d)
      }
      .mergeSubstreams
  }

  def parseParameters(vDim: Int, wDim: Int)(
      ps: Vector[String]): Option[(Int, String, DlmParameters)] = {
    for {
      sId <- ps(0).toIntOpt
      fcstTime = ps(1)
      v = ps.drop(2).take(vDim).map(_.toDouble).toArray
      w = ps.drop(2 + vDim).take(wDim).map(_.toDouble).toArray
      m0 = ps.drop(2 + 2 * wDim).take(wDim).map(_.toDouble).toArray
      c0 = ps.drop(2 + 3 * wDim).take(wDim).map(_.toDouble).toArray
    } yield
      (sId,
       fcstTime,
       DlmParameters(
         diag(DenseVector(v)),
         diag(DenseVector(w)),
         DenseVector(m0),
         diag(DenseVector(c0))
       ))
  }
}

/**
  * Determine V and W parameters independently for forecast hours 1 to 18 for each
  */
object FirstOrderRadiance extends App with ReadRadianceDifference {
  implicit val system = ActorSystem("NationalGrid")
  implicit val materializer = ActorMaterializer()

  val model = List.fill(18)(Dlm.polynomial(1)).reduce(_ |*| _)
  val initP = DlmParameters(
    v = diag(DenseVector.fill(18)(1.0)),
    w = diag(DenseVector.fill(18)(1.0)),
    m0 = DenseVector.fill(18)(0.0),
    c0 = diag(DenseVector.fill(18)(10.0))
  )

  val priorV = InverseGamma(3.0, 5.0)
  val priorW = InverseGamma(10.0, 10.0)

  def mcmc(data: Vector[Data]) =
    GibbsSampling.sample(model, priorV, priorW, initP, data).steps.take(10000)

  def formatParameters(p: DlmParameters): List[Double] =
    p.toList

  readRadiance("examples/data/radiance_difference_training.csv")
    .collect { case Some(a) => a }
    .via(radianceToData)
    .mapAsyncUnordered(4) {
      case (stationId, fcstTime, data) =>
        println(s"Station ID: $stationId")
        println(s"Forecast Time: $fcstTime")
        println(s"Data Length: ${data.size}")

        val iters = Source.fromIterator(() => mcmc(data.sortBy(_.time)))

        val params: Future[DlmParameters] = iters
          .alsoTo(
            Streaming.writeChainSink(
              s"examples/data/station_${stationId}_${fcstTime}_first_order.csv",
              (s: GibbsSampling.State) => formatParameters(s.p)))
          .map(_.p)
          .via(Streaming.meanParameters(18, 18))
          .runWith(Sink.head)

        val res: Future[(Int, Int, DlmParameters)] = for {
          p <- params
        } yield (stationId, fcstTime, DlmParameters(p.v, p.w, p.m0, p.c0))

        Source
          .fromFuture(res)
          .runWith(
            writeMeanParams(formatParameters,
                            "examples/data/first_order_mean_radiance.csv"))
    }
    .runWith(Sink.onComplete { s =>
      system.terminate()
    })
}

object ForecastFirstOrderRadiance extends App with ReadRadianceDifference {

  implicit val system = ActorSystem("NationalGrid")
  implicit val materializer = ActorMaterializer()
  val format = DateTimeFormatter.ISO_DATE_TIME
  implicit val codec: CellCodec[LocalDateTime] = localDateTimeCodec(format)

  val model = List.fill(18)(Dlm.polynomial(1)).reduce(_ |*| _)

  def meanSquaredError(
      forecast: Vector[(Double, DenseVector[Double], DenseMatrix[Double])],
      testData: Vector[Data]) = {

    val ss = (forecast zip testData).map {
      case (f, x) =>
        val c = f._2 - x.observation.map(a => a.getOrElse(0.0))
        c dot c
    }.sum

    ss / forecast.size
  }

  // convert hours from the epoch to datetime
  def hoursToDatetime(hours: Double) = {
    val inst = Instant.ofEpochSecond(hours.toLong * 60 * 60)
    val tz = ZoneOffset.UTC
    LocalDateTime.ofInstant(inst, tz)
  }

  readRadiance("examples/data/radiance_difference_test.csv")
    .collect { case Some(a) => a }
    .via(radianceToData)
    .groupBy(1000, a => (a._1, a._2))
    .mapAsync(1) {
      case (stationId, fcstTime, data) =>
        println(s"Getting file for station $stationId")
        val paramFile = s"examples/data/station_${stationId}_2_first_order.csv"

        val out = new java.io.File(
          s"examples/data/forecast_radiance_${stationId}_fo.csv")
        val headers = rfc.withHeader(false)

        val times = data.map(x => hoursToDatetime(x.time))

        for {
          p <- Streaming
            .readCsv(paramFile)
            .map(_.map(_.toDouble).toList)
            .drop(1000)
            .map(
              a =>
                DlmParameters(
                  v = diag(DenseVector(a.take(18).toArray)),
                  w = diag(DenseVector(a.drop(18).toArray)),
                  m0 = DenseVector.zeros[Double](18),
                  c0 = DenseMatrix.eye[Double](18) * 100.0
              ))
            .via(Streaming.meanParameters(18, 18))
            .runWith(Sink.head)
          training <- readRadiance(
            "examples/data/radiance_difference_training.csv")
            .collect { case Some(a) => a }
            .via(radianceToData)
            .filter(_._1 == stationId)
            .map(_._3)
            .runWith(Sink.head)
          kf = KalmanFilter(KalmanFilter.advanceState(p, model.g))
          init = kf.initialiseState(model, p, training)
          lastState = training.foldLeft(init)(kf.step(model, p))
          newP = p.copy(m0 = lastState.mt, c0 = lastState.ct)
          filtered = kf.filter(model, data, newP)
          summary = filtered.flatMap(x =>
            (for {
              f <- x.ft
              q <- x.qt
            } yield Dlm.summariseForecast(0.75)(f, q)).map(f =>
              f.flatten.toList))
          io = out.writeCsv(times.zip(summary), headers)
        } yield io
    }
    .mergeSubstreams
    .runWith(Sink.onComplete { s =>
      println(s)
      system.terminate()
    })
}
