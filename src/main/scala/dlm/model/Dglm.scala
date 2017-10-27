package dlm.model

import breeze.stats.distributions._
import breeze.linalg._
import cats.implicits._
import math.exp

/**
  * DGLM
  */
object Dglm {
  /**
    * A class representing a DGLM
    */
  case class Model(
    observation: (DenseVector[Double], DenseMatrix[Double]) => Rand[DenseVector[Double]],
    f: ObservationMatrix,
    g: SystemMatrix,
    conditionalLikelihood: (Dlm.Parameters) => ConditionalLl
  )

  /**
    * Logistic function to transform the number onto a range between 0 and upper
    * @param upper the upper limit of the logistic function
    * @param number the number to be transformed
    * @return a number between 0 and upper
    */
  def logisticFunction(upper: Double)(number: Double) = {
    if (number < -5) {
      0.0
    } else if (number > 5) {
      upper
    } else {
      upper / (1 + math.exp(-number))
    }
  }

  /**
    * Construct a DGLM with Beta distributed observations, 
    * with variance < mean (1 - mean)
    */
  def beta(mod: Dlm.Model): Dglm.Model = {

    Dglm.Model(
      observation = (x, v) => {
        val mean = logisticFunction(1.0)(x(0))
        val a = (mean * (1 - mean)) / v(0,0)
        val alpha = mean * (a - 1)
        val beta = (1 - mean) * (a - 1)
        
        new Beta(alpha, beta).map(DenseVector(_))
      },
      f = mod.f,
      g = mod.g,
      conditionalLikelihood = p => (x, y) => {
        val mean = logisticFunction(1.0)(x(0))
        val a = (mean * (1 - mean)) / p.v(0,0)
        val alpha = mean * (a - 1)
        val beta = (1 - mean) * (a - 1)

        new Beta(alpha, beta).logPdf(y(0))
      })
  }

  /**
    * Construct a DGLM with Poisson distributed observations
    */
  def poisson(mod: Dlm.Model): Dglm.Model = {
    Dglm.Model(
      observation = (x, v) => Poisson(exp(x(0))).map(DenseVector(_)),
      f = mod.f,
      g = mod.g,
      conditionalLikelihood = p => (x, y) =>
      Poisson(exp(x(0))).logProbabilityOf(y(0).toInt)
    )
  }

  def simStep(
    mod: Model, 
    p: Dlm.Parameters) = (time: Time, x: DenseVector[Double]) => {
    for {
      x1 <- MultivariateGaussianSvd(mod.g(time) * x, p.w)
      y <- mod.observation(mod.f(time).t * x1, p.v)
    } yield (Data(time + 1, y.some), x1)
  }

  def simulate(mod: Model, p: Dlm.Parameters) = {
    val initState = (Data(0, None), MultivariateGaussianSvd(p.m0, p.c0).draw)
    MarkovChain(initState){ case (d, x) => simStep(mod, p)(d.time, x) }
  }
}
