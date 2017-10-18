package dlm.examples.urbanobservatory

import dlm.model._
import Dlm._
import MetropolisHastings._
import cats.implicits._
import breeze.stats.distributions.{Beta, MarkovChain, Gaussian}
import breeze.linalg.{DenseVector, DenseMatrix, diag}
import kantan.csv._
import kantan.csv.ops._

/**
  * Beta DGLM for Humidity data
  */
object HumidityDglm extends App with ObservedData {
  val mod = Dlm.polynomial(1) |+| Dlm.seasonal(24, 3)
  val humidityData = data.
    map(d => Data(d.time, d.observation.map(x => DenseVector(x(0)))))

  def likelihood(parameters: (Double, Parameters)) = 
    ParticleFilter.likelihood(mod, humidityData, parameters._2, 1000, Dglm.beta(parameters._1))

  def proposal(parameters: (Double, Parameters)) = {
    for {
      newVar <- Gaussian(0.0, 0.5).map(i => parameters._1 * math.exp(i))
      newP <- symmetricProposal(0.05)(parameters._2)
    } yield (newVar, newP)
  }

  val step = mhStep[(Double, Parameters)](proposal, likelihood) _

  val initP = (0.5, Parameters(
    v = DenseMatrix((1e-10)),
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
  ))

  val init = MhState[(Double, Parameters)](initP, -1e99, 0)
  val iters = MarkovChain(init)(step).steps.take(100000)

  val out = new java.io.File("data/humidity_dglm_parameters.csv")
  val headers = rfc.withHeader("variance", "W1", "W2", "W3", "W4", "W5", "W6", "W7")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatParameters(p: (Double, Parameters)) = {
    val w = diag(p._2.w).data

    (w :+ p._1).toList
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.parameters))
  }

  writer.close()

}
