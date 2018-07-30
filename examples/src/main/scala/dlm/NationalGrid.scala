package examples.dlm

import breeze.linalg.{DenseVector, DenseMatrix, diag}
import core.dlm.model._
import akka.actor.ActorSystem
import java.time._
import java.time.format._
import breeze.linalg.{DenseVector, diag}
import cats.implicits._
import scala.math.Ordering
import java.nio.file.Paths
import akka.stream.scaladsl._
import akka.util.ByteString
import akka.NotUsed
import StringUtils._

import scala.concurrent.ExecutionContext.Implicits.global
import akka.stream._
import akka.stream.scaladsl._
import scala.concurrent.Future

/**
  * Determine V and W parameters independently for forecast hours 1 to 18 for each 
  */
object FirstOrderRadiance extends App {
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
  val priorW = InverseGamma(3.0, 5.0)

  def mcmc(data: Vector[Dlm.Data]) = 
    GibbsSampling.sample(model, priorV, priorW, initP, data).
      steps.
      take(10000)

  def formatParameters(p: DlmParameters): List[Double] = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  WeatherData.readRadiance("/Users/jonny/Documents/PhD/thesis-scala-examples/data/radiance_cleaned_training.csv").
    collect { case Some(a) => a }.
    filter(_.stationId == 41).
    via(WeatherData.radianceToDataFlow).
    mapAsyncUnordered(2) { case (genId, data) =>

      println(s"Gen ID: $genId")
      println(s"Data Length: ${data.size}")

      val iters = Source.fromIterator(() => mcmc(data.sortBy(_.time)))

      val params: Future[DlmParameters] = iters.
        map(_.p).
        alsoTo(Write.writeChain(formatParameters, s"/Users/jonny/Desktop/station_${genId}_first_order.csv")).
        via(Forecasting.meanParameters(18)).
        runWith(Sink.head)

      val state: Future[(DenseVector[Double], DenseMatrix[Double])] = iters.
        map(_.state.last._2).
        runWith(Sink.seq).
        map(Dglm.meanCovSamples)

      val res: Future[(Double, DlmParameters)] = for {
        p <- params
        st <- state

      } yield (genId, DlmParameters(p.v, p.w, st._1, st._2))

      res
    }.
    runWith(Sink.onComplete { s =>
      system.terminate()
    })
}

// object Forecast {
//   val params = readMeanParams("data/output/first_order_mean_radiance.csv").
//     collect { case Some(a) => a }

//   val data = readRadiance("data/radiance_cleaned_test.csv").
//     collect { case Some(a) => a }.
//     filter(_.stationId == 41).
//     via(radianceToDataFlow)

//   (params zip data).
//     groupBy(_.stationId)
//     mapAsyncUnordered(2) { case (p, (genId, data)) =>

//       val fcstError = for {
//         p <- params
//         d = trainingData.filter(_._1 == genId)
//         f = forecastRadiance(model, p, mt, ct, d)
//       } yield meanSquaredError(f, d)
//     }
// }

object Forecasting {
  def zeroParameters(dim: Int): DlmParameters = {
    DlmParameters(
      v = DenseMatrix.zeros[Double](dim, dim),
      w = DenseMatrix.zeros[Double](dim, dim),
      m0 = DenseVector.zeros[Double](dim),
      c0 = DenseMatrix.zeros[Double](dim, dim))
  }

  def add(x: DlmParameters, y: DlmParameters): DlmParameters = {
    DlmParameters(x.v + y.v, x.w + y.w, x.m0 + y.m0, x.c0 + y.c0)
  }

  /**
    * Calculate the streaming mean of parameters
    */
  def meanParameters(dim: Int) = {
    Flow[DlmParameters].
      fold((zeroParameters(dim), 1.0))((acc, b) => {
        val (avg: DlmParameters, n: Double) = acc
        println(s"current average $avg")
        println(s"new params $b")
        (add(avg.map(_ * n), b).map(_  / (n + 1)), n + 1)
      }).
      map(_._1)
  }

  def meanState(dim: Int) = {
    Flow[DenseVector[Double]].
      fold((DenseVector.zeros[Double](dim), 1.0))((acc, b) => {
        val (avg, n) = acc
        ((avg * n + b)  / (n + 1), n + 1)
      }).
      map(_._1)
  }

  /**
    * Perform 18 step forecast for the final day radiance
    */
  def forecastRadiance(
    model: core.dlm.model.DlmModel,
    p: DlmParameters,
    mt: DenseVector[Double],
    ct: DenseMatrix[Double],
    time: Double) = {

    Dlm.forecast(model, mt, ct, time, p).
      take(18).
      toVector
  }

  /**
    * Calculate the mean squared error
    */
  def meanSquaredError(
    forecast: Vector[(Double, Double, Double)],
    testData: Vector[(Double, Double)]) = {

    val ss = (forecast zip testData).
      map { case (f, x) => (f._2 - x._2) * (f._2 - x._2) }.
      sum

    ss / forecast.size
  }
}

object WeatherData {
  /**
    * An ordering for local Date Time
    */
  implicit def orderDateTime: Ordering[LocalDateTime] = 
    Ordering.by(_.toEpochSecond(ZoneOffset.UTC))
  
  case class Radiance(
    stationId:        Int, 
    forecastDatetime: LocalDateTime,
    difference:       Vector[Option[Double]]
  )

  /**
    * Parse a String to a radiance
    */
  def parseRadiance(l: Vector[String]): Option[Radiance] = {
    val format = DateTimeFormatter.ISO_DATE_TIME
    for {
      sId <- l(0).toIntOpt
      fcDatetime <- l(1).toDatetimeOpt(format)
      diff = l.drop(2).map(_.toDoubleOpt)
    } yield Radiance(sId, fcDatetime, diff)
  }


  def readRadiance(file: String) = {
    FileIO.fromPath(Paths.get(file)).
      via(Framing.delimiter(ByteString("\n"), 
        maximumFrameLength = 8192, allowTruncation = true)).
      map(_.utf8String).
      map(a => a.split(",").toVector).
      map(parseRadiance)
  }

  def datetimeToSeconds(s: LocalDateTime): Long = {
    s.toEpochSecond(ZoneOffset.UTC)
  }


  def secondsToDatetime(seconds: Long): LocalDateTime = {
    Instant.ofEpochSecond(seconds).atZone(ZoneOffset.UTC).toLocalDateTime()
  }

  /**
    * Convert a single radiance reading into a forecast datetime
    * lead time and difference
    */
  def radianceToData(r: Radiance): (Int, Dlm.Data) = 
    (r.stationId, 
      Dlm.Data(
        datetimeToSeconds(r.forecastDatetime), 
        DenseVector(r.difference.toArray)
      )
    )

  /**
    * Given a single station ID, summarise the difference between
    * forecast and actual for each lead time
    * @return a map from lead time and forecast datetime 
    * to a vector of data containing the forecast datetime and difference
    * at each lead time
    */
  def radianceToDataFlow: Flow[Radiance, (Double, Vector[Dlm.Data]), NotUsed] = {
    Flow[Radiance].
      groupBy(100, a => a.stationId).
      fold((0.0, Vector.empty[Dlm.Data])){(l, r) =>
        val (lt, d) = radianceToData(r)
        (lt, l._2 :+ d)
      }.
      mergeSubstreams
  }

  def parseParameters(dim: Int)(ps: Vector[String]): Option[(Int, DlmParameters)] = {
    for {
      sId <- ps(0).toIntOpt
      v = ps.drop(1).take(dim).map(_.toDouble).toArray
      w = ps.drop(1 + dim).take(dim).map(_.toDouble).toArray
      m0 = ps.drop(1 + 2 * dim).take(dim).map(_.toDouble).toArray
      c0 = ps.drop(1 + 3 * dim).take(dim).map(_.toDouble).toArray
    } yield (sId, DlmParameters(
      diag(DenseVector(v)),
      diag(DenseVector(w)),
      DenseVector(m0),
      diag(DenseVector(c0))
    ))
  }

  def readMeanParams(file: String) = {
    FileIO.fromPath(Paths.get(file)).
      via(Framing.delimiter(ByteString("\n"), 
        maximumFrameLength = 8192, allowTruncation = true)).
      map(_.utf8String).
      map(a => a.split(",").toVector).
      map(parseParameters(18))
  }
}

object Write {
  def writeChain(
    formatParameters: DlmParameters => List[Double],
    file: String): Sink[DlmParameters, Future[IOResult]] = {

    Flow[DlmParameters].
      map(p => formatParameters(p).mkString(", ")).
      map(s => ByteString(s + "\n")).
      toMat(FileIO.toPath(Paths.get(file)))(Keep.right)
  }

  def writeMeanParams(
    formatParameters: DlmParameters => List[Double],
    file: String) = {

    Flow[(Int, DlmParameters)].
      map{ case (id, p) => (id :: formatParameters(p)).mkString(", ") }.
      map(s => ByteString(s + "\n")).
      toMat(FileIO.toPath(Paths.get(file)))(Keep.right)
  }
}

object StringUtils {
  implicit class StringImprovements(val s: String) {
    import scala.util.control.Exception._
    def toIntOpt = 
      catching(classOf[NumberFormatException]) opt s.toInt

    def toDatetimeOpt(format: DateTimeFormatter) = 
      catching(classOf[DateTimeException]) opt LocalDateTime.parse(s, format)

    def toDoubleOpt = 
      catching(classOf[NumberFormatException]) opt s.toDouble
  }
}
