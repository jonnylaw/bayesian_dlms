package dlm.examples.urbanobservatory

import dlm.model._
import Dlm._

import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.{Gamma, Gaussian, Rand}
import breeze.numerics.exp

import cats.implicits._
import cats.Applicative

import java.nio.file.Paths
import kantan.csv._
import kantan.csv.ops._

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

  val iters = GibbsSampling.gibbsSamples(temperatureModel, InverseGamma(1.0, 1.0), InverseGamma(1.0, 1.0), initP, temperatureData).
    steps.
    take(1000000)

  val out = new java.io.File("data/temperature_model_parameters_gibbs.csv")
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

object FitTemperatureModelMetrop extends App with Models with ObservedData {
  val temperatureData = data.
    map(d => Data(d.time, d.observation.map(x => DenseVector(x(1)))))

  def proposal(delta: Double) = { p: Parameters =>
    for {
      innov_w <- Applicative[Rand].replicateA(7, Gaussian(0, delta))
      w = diag(p.w) *:* DenseVector(exp(innov_w.toArray))
      innov_v <- Gaussian(0, delta)
      v = diag(p.v) *:* DenseVector(exp(innov_v))
    } yield Parameters(diag(v), diag(w), p.m0, p.c0)
  }

  def formatParameters(p: Parameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  val iters: Iterator[Parameters] = MetropolisHastings.metropolisHastingsDlm(
      temperatureModel, temperatureData, proposal(0.05), initP).
      steps.
      map(_.parameters).
      take(1000000)

  Streaming.writeChain(
    formatParameters, "data/temperature_model_parameters.csv",
    rfc.withHeader("V", "W1", "W2", "W3", "W4", "W5", "W6", "W7"))(iters)
}

object FitHumidityModel extends App with Models with ObservedData {
  val humidityData = data.
    map(d => Data(d.time, d.observation.map(x => DenseVector(x(0)))))

  def proposal(delta: Double) = { p: Parameters =>
    for {
      innov_w <- Applicative[Rand].replicateA(7, Gaussian(0, delta))
      w = diag(p.w) *:* DenseVector(exp(innov_w.toArray))
      innov_v <- Gaussian(0, delta)
      v = diag(p.v) *:* DenseVector(exp(innov_v))
    } yield Parameters(diag(v), diag(w), p.m0, p.c0)
  }

  def formatParameters(p: Parameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  val iters: Iterator[Parameters] = MetropolisHastings.metropolisHastingsDlm(
      humidityModel, humidityData, proposal(0.05), initP).
      steps.
      map(_.parameters).
      take(1000000)

  Streaming.writeChain(
    formatParameters, "data/humidity_model_parameters_metrop.csv",
    rfc.withHeader("V", "W1", "W2", "W3", "W4", "W5", "W6", "W7"))(iters)
}

object FitHumidityModelGibbs extends App with Models with ObservedData {
  val humidityData = data.
    map(d => Data(d.time, d.observation.map(x => DenseVector(x(0)))))

  val iters = GibbsSampling.gibbsSamples(humidityModel, InverseGamma(1.0, 1.0), InverseGamma(1.0, 1.0), initP, humidityData).
    steps.
    take(1000000)

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
