package dlm.examples

import dlm.model._
import Dlm._
import cats.implicits._
import cats.Applicative
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.{Gamma, Gaussian, Rand}
import java.nio.file.Paths
import breeze.numerics.exp
import kantan.csv._
import kantan.csv.ops._
import kantan.csv.generic._

trait Models {
  val temperatureModel = Dlm.polynomial(1) |+| Dlm.seasonal(24, 3)
  val humidityModel = temperatureModel

  val initP = Dlm.Parameters(
    v = DenseMatrix((1.906964895e-18)),
    w = diag(DenseVector(
      1.183656514e-01,
      4.074187352e-13,
      4.162722546e-01, 
      7.306081843e-14, 
      2.193511341e-03, 
      1.669158400e-08, 
      3.555685730e-03)),
    m0 = DenseVector.fill(7)(0.0),
    c0 = diag(DenseVector.fill(7)(1.0))
  )
}

trait ObservedData {
  val rawData = Paths.get("data/humidity_temperature_1114.csv")
  val reader = rawData.asCsvReader[(Time, Double, Double)](rfc.withHeader)
  val data = reader.
    collect { 
      case Success(a) => if (a._2.isNaN || a._3.isNaN) {
        Data(a._1, None)
      } else {
        Data(a._1, DenseVector(a._2, a._3).some)
      }
    }.
    toArray  
}

object FitTemperatureModel extends App with Models with ObservedData {
  val temperatureData = data.
    map(d => Data(d.time, d.observation.map(x => DenseVector(x(1)))))

  def proposal(delta: Double) = { p: Parameters =>
    for {
      m <- p.m0.data.toVector traverse (x => Gaussian(x, delta): Rand[Double])
      innov <- Applicative[Rand].replicateA(7, Gaussian(0, delta))
      c = diag(p.c0) + DenseVector(exp(innov.toArray))
    } yield Parameters(p.v, p.w, DenseVector(m.toArray), diag(c))
  }

  val iters = GibbsSampling.gibbsSamples(temperatureModel, Gamma(1.0, 1.0), Gamma(1.0, 1.0), initP, temperatureData).
    steps.
    take(100000)

  val out = new java.io.File("data/temperature_model_parameters.csv")
  val headers = rfc.withHeader("V", "W1", "W2", "W3", "W4", "W5", "W6", "W7")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatParameters(p: Parameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.p))
  }

  writer.close()
}

object FitHumidityModel extends App with Models with ObservedData {
  val humidityData = data.
    map(d => Data(d.time, d.observation.map(x => DenseVector(x(0)))))

  def proposal(delta: Double) = { p: Parameters =>
    for {
      m <- p.m0.data.toVector traverse (x => Gaussian(x, delta): Rand[Double])
      innov <- Applicative[Rand].replicateA(7, Gaussian(0, delta))
      c = diag(p.c0) + DenseVector(exp(innov.toArray))
    } yield Parameters(p.v, p.w, DenseVector(m.toArray), diag(c))
  }

  val iters = GibbsSampling.gibbsSamples(humidityModel, Gamma(1.0, 1.0), Gamma(1.0, 1.0), initP, humidityData).
    steps.
    take(100000)

  val out = new java.io.File("data/humidity_model_parameters.csv")
  val headers = rfc.withHeader("V", "W1", "W2", "W3", "W4", "W5", "W6", "W7")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatParameters(p: Parameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.p))
  }

  writer.close()
}

object JointModel extends App with Models with ObservedData {
  val combinedModel = Dlm.outerSumModel(temperatureModel, humidityModel)
  val combinedParameters = Dlm.outerSumParameters(initP, initP)

  val iters = GibbsWishart.gibbsSamples(humidityModel, Gamma(1.0, 1.0), InverseWishart(15.0, DenseMatrix.eye[Double](14)), initP, data).
    steps.
    take(100000)

  val out = new java.io.File("data/humidity_temp_model_parameters.csv")
  val headers = rfc.withHeader("V", "W1", "W2", "W3", "W4", "W5", "W6", "W7")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatParameters(p: Parameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.p))
  }

  writer.close()

}
