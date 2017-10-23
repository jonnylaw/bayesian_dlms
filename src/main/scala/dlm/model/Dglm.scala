package dlm.model

import breeze.stats.distributions._
import breeze.linalg._

/**
  *  Univariate DGLM
  */
object Dglm {
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
    * Conditional Likelihood for Beta distributed observations with variance < mean (1 - mean)
    */
  def beta(variance: Double)(y: Observation, state: DenseVector[Double]) = {
    val mean = logisticFunction(1.0)(state(0))
    val a = (mean * (1 - mean)) / variance
    val alpha = mean * (a - 1)
    val beta = (1 - mean) * (a - 1)
    new Beta(alpha, beta).logPdf(y(0))
  }

  /**
    * Conditional Likelihood for Poisson distributed observations
    */
  def poisson(y: Observation, state: DenseVector[Double]) = 
    Poisson(state(0)).logProbabilityOf(y(0).toInt)
}
