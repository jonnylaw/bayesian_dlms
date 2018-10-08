package examples.dlm

import breeze.linalg.{DenseVector, DenseMatrix, diag}
import dlm.core.model._
import akka.actor.ActorSystem
import cats.implicits._
import scala.concurrent.ExecutionContext.Implicits.global
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
    stationId:        Int, 
    forecastDatetime: LocalDateTime,
    forecastHour:     Int,
    difference:       Vector[Option[Double]]
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

    Flow[(Int, Int, DlmParameters)].
      map{ case (id, fsctTime, p) => (id :: fsctTime :: formatParameters(p)).mkString(", ") }.
      map(s => ByteString(s + "\n")).
      toMat(FileIO.toPath(Paths.get(file)))(Keep.right)
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
    FileIO.fromPath(Paths.get(file)).
      via(Framing.delimiter(ByteString("\n"), 
        maximumFrameLength = 8192, allowTruncation = true)).
      map(_.utf8String).
      map(a => a.split(",").toVector).
      map(parseRadiance)
  }

  /**
    * Convert a single radiance reading into a forecast datetime
    * lead time and difference
    */
  def radianceToData(r: Radiance): (Int, Int, Data) = 
    (r.stationId, 
      r.forecastHour,
      Data(r.forecastDatetime.toEpochSecond(ZoneOffset.UTC),
        DenseVector(r.difference.toArray))
    )

  /**
    * Given a single station ID, summarise the difference between
    * forecast and actual for each lead time
    * @return a map from lead time and forecast datetime 
    * to a vector of data containing the forecast datetime and difference
    * at each lead time
    */
  def radianceToData: Flow[Radiance, (Int, Int, Vector[Data]), NotUsed] = {
    Flow[Radiance].
      groupBy(100, a => a.stationId).
      fold((0, 0, Vector.empty[Data])){(l, r) =>
        val (lt, forecastTime, d) = radianceToData(r)
        (lt, forecastTime, l._3 :+ d)
      }.
      mergeSubstreams
  }

  def parseParameters(vDim: Int, wDim: Int)(ps: Vector[String]): Option[(Int, String, DlmParameters)] = {
    for {
      sId <- ps(0).toIntOpt
      fcstTime = ps(1)
      v = ps.drop(2).take(vDim).map(_.toDouble).toArray
      w = ps.drop(2 + vDim).take(wDim).map(_.toDouble).toArray
      m0 = ps.drop(2 + 2 * wDim).take(wDim).map(_.toDouble).toArray
      c0 = ps.drop(2 + 3 * wDim).take(wDim).map(_.toDouble).toArray
    } yield (sId, fcstTime, DlmParameters(
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

  val model = List.fill(18)(Dlm.polynomial(1)).
    reduce(_ |*| _)
  val initP = DlmParameters(
    v = diag(DenseVector.fill(18)(1.0)),
    w = diag(DenseVector.fill(18)(1.0)),
    m0 = DenseVector.fill(18)(0.0),
    c0 = diag(DenseVector.fill(18)(10.0))
  )

  val priorV = InverseGamma(3.0, 5.0)
  val priorW = InverseGamma(10.0, 10.0)

  def mcmc(data: Vector[Data]) = 
    GibbsSampling.sample(model, priorV, priorW, initP, data).
      steps.
      take(10000)

  def formatParameters(p: DlmParameters): List[Double] = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  readRadiance("examples/data/radiance_difference_training.csv").
    collect { case Some(a) => a }.
    via(radianceToData).
    mapAsyncUnordered(4) { case (stationId, fcstTime, data) =>

      println(s"Station ID: $stationId")
      println(s"Forecast Time: $fcstTime")
      println(s"Data Length: ${data.size}")

      val iters = Source.fromIterator(() => mcmc(data.sortBy(_.time)))

      val params: Future[DlmParameters] = iters.
        alsoTo(Streaming.writeChainSink(s"examples/data/station_${stationId}_${fcstTime}_first_order.csv",
          (s: GibbsSampling.State) => formatParameters(s.p))).
        map(_.p).
        via(Streaming.meanParameters(18, 18)).
        runWith(Sink.head)

      val state: Future[(DenseVector[Double], DenseMatrix[Double])] = iters.
        map(_.state.last.sample).
        runWith(Sink.seq).
        map(Dglm.meanCovSamples)

      val res: Future[(Int, Int, DlmParameters)] = for {
        p <- params
        st <- state
      } yield (stationId, fcstTime, DlmParameters(p.v, p.w, st._1, st._2))

      Source.fromFuture(res).
        runWith(writeMeanParams(formatParameters, "examples/data/first_order_mean_radiance.csv"))
    }.
    runWith(Sink.onComplete { s =>
      system.terminate()
    })
}

object ForecastFirstOrderRadiance extends App
    with ReadRadianceDifference {

  implicit val system = ActorSystem("NationalGrid")
  implicit val materializer = ActorMaterializer()

  val model = List.fill(18)(Dlm.polynomial(1)).
    reduce(_ |*| _)

  def meanSquaredError(
    forecast: Vector[(Double, DenseVector[Double], DenseMatrix[Double])],
    testData: Vector[Data]) = {

    val ss = (forecast zip testData).
      map { case (f, x) =>
        val c = f._2 - x.observation.map(a => a.getOrElse(0.0))
        c dot c }.
      sum

    ss / forecast.size
  }

  val params = scala.io.Source.
    fromFile(new java.io.File("examples/data/first_order_mean_radiance.csv")).
    getLines.
    map(a => a.split(",").toVector).
    map(parseParameters(18, 18)).
    collect { case Some(a) => a }

  readRadiance("examples/data/radiance_difference_test.csv").
    collect { case Some(a) => a }.
    filter(_.stationId == 41).
    via(radianceToData).
    groupBy(100, a => (a._1, a._2)).
    map { case (stationId, fcstTime, data) =>

      // extract the parameters for the appropriate station
      // and forecast datetime
      val p = params.
        filter(x => x._1 == stationId & x._2 == fcstTime).
        map(_._3).toVector.head

      // Perform an 18 hour forecast
      val f = Dlm.forecast(model, p.m0, p.c0, data.head.time, p).
        take(18).
        toVector

      meanSquaredError(f, data)
    }.
    mergeSubstreams.
    runForeach(println).
    onComplete(_ => system.terminate())
}
