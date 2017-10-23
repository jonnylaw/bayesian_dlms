package dlm.examples

import dlm.model._
import GibbsSampling._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Poisson
import java.nio.file.Paths
import java.io.File
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._

trait PoissonDglm {
  val mod = Dglm.Model(
    observation = (x, v) => Poisson(x(0)).map(DenseVector(_)),
    f = (t: Time) => DenseMatrix((1.0)), 
    g = (t: Time) => DenseMatrix((1.0))
  )
  val p = Dlm.Parameters(
    DenseMatrix(3.0), 
    DenseMatrix(1.0), 
    DenseVector(0.0), 
    DenseMatrix(1.0))
}

trait PoissonData {
  val rawData = Paths.get("data/poisson_dglm.csv")
  val reader = rawData.asCsvReader[(Time, Double, Double)](rfc.withHeader)
  val data = reader.
    collect { 
      case Success(a) => Dlm.Data(a._1, Some(a._2).map(DenseVector(_)))
    }.
    toArray
}

object SimulatePoissonDglm extends App with PoissonDglm {
  val sims = Dglm.simulate(mod, p).
    steps.
    take(100)

  val out = new java.io.File("data/poisson_dglm.csv")
  val writer = out.asCsvWriter[(Time, Option[Double], Double)](rfc.withHeader("time", "observation", "state"))

  def formatData(d: (Dlm.Data, DenseVector[Double])) = d match {
    case (Dlm.Data(t, y), x) =>
      (t, y.map(x => x(0)), x(0))
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

/**
  * Use Particle Gibbs to determine the parameters of the poisson DGLM
  */
object PoissonDglmGibbs extends App with PoissonDglm with PoissonData {

}
