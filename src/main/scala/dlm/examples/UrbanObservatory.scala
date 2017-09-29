package dlm.examples

import dlm.model._
import Dlm._
import cats.implicits._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.Gamma
import java.nio.file.Paths
import kantan.csv._
import kantan.csv.ops._
import kantan.csv.generic._

trait Models {
  val temperatureModel = Dlm.polynomial(1) |+| Dlm.seasonal(24, 3)

  val initP = Dlm.Parameters(
    v = DenseMatrix((1.0)),
    w = diag(DenseVector.fill(7)(1.0)),
    m0 = DenseVector.fill(7)(0.0),
    c0 = diag(DenseVector.fill(7)(10.0))
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

object SimulateTemperature extends App with Models {
  val sims = simulate(0, temperatureModel, initP).
    steps.
    take(1000)

  val out = new java.io.File("data/simulated_temperature_data.csv")
  val writer = out.asCsvWriter[(Time, Option[Double], Double, Double)](rfc.withHeader("time", "observation", "state1", "state"))

  def formatData(d: (Data, DenseVector[Double])) = d match {
    case (Data(t, y), state) =>
      (t, y.map(x => x(0)), state(0), state(1))
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

object FitTemperatureModel extends App with Models with ObservedData {
  val temperatureData = data.map(d => Data(d.time, d.observation.map(x => DenseVector(x(1)))))

  val iters = GibbsSampling.gibbsSamples(temperatureModel, Gamma(1.0, 1.0), Gamma(1.0, 1.0), initP, temperatureData).
    steps.
    take(10)

  iters.map(x => formatParameters(x.p)).foreach(println)

  // val out = new java.io.File("data/temperature_model_parameters.csv")
  // val headers = rfc.withHeader("V", "W1", "W2", "W3", "W4", "W5", "W6", "W7")
  // val writer = out.asCsvWriter[List[Double]](headers)

  def formatParameters(p: Parameters) = {
    (p.v.data ++ p.w.data).toList
  }

  // // write iters to file
  // while (iters.hasNext) {
  //   writer.write(formatParameters(iters.next.p))
  // }

  // writer.close()
}
