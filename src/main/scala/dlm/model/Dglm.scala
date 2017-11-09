package dlm.model

import breeze.stats.distributions._
import breeze.linalg.{DenseVector, DenseMatrix}
import cats.implicits._
import math.exp
import breeze.stats.covmat

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

  def studentT(df: Int, mod: Dlm.Model): Dglm.Model = {
    Dglm.Model(
      observation = (x: DenseVector[Double], v: DenseMatrix[Double]) =>
        StudentsT(df).map(a => DenseVector(a * v(0,0) + x(0))),
      mod.f,
      mod.g,
      conditionalLikelihood = (p: Dlm.Parameters) => 
      (y: Observation, x: DenseVector[Double]) => 
      1/p.v(0,0) * StudentsT(df).logPdf((y(0) - x(0)) / p.v(0,0))
    )
  }

  def beta(mean: Double, variance: Double): ContinuousDistr[Double] = {
    val a = (mean * (1 - mean)) / variance
    val alpha = mean * (a - 1)
    val beta = (1 - mean) * (a - 1)
    new Beta(alpha, beta)
  }

  /**
    * Construct a DGLM with Beta distributed observations, 
    * with variance < mean (1 - mean)
    */
  def beta(mod: Dlm.Model): Dglm.Model = {

    Dglm.Model(
      observation = (x, v) => {
        val mean = logisticFunction(1.0)(x(0))
        
        beta(mean, v(0,0)).map(DenseVector(_))
      },
      f = mod.f,
      g = mod.g,
      conditionalLikelihood = p => (x, y) => {
        val mean = logisticFunction(1.0)(x(0))

        beta(mean, p.v(0, 0)).logPdf(y(0))
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
    p:   Dlm.Parameters) = (time: Time, x: DenseVector[Double]) => {
    for {
      x1 <- MultivariateGaussianSvd(mod.g(time) * x, p.w)
      y <- mod.observation(mod.f(time).t * x1, p.v)
    } yield (Data(time + 1, y.some), x1)
  }

  def simulate(mod: Model, p: Dlm.Parameters) = {
    val initState = (Data(0, None), MultivariateGaussianSvd(p.m0, p.c0).draw)
    MarkovChain(initState){ case (d, x) => simStep(mod, p)(d.time, x) }
  }

 /**
    * Calculate the mean and covariance of a sequence of DenseVectors
    */
  def meanCovSamples(samples: Seq[DenseVector[Double]]) = {
    val n = samples.size
    val m = new DenseMatrix(n, samples.head.size, 
      samples.map(_.data).toArray.transpose.flatten)
    val sampleMean = samples.reduce(_ + _).map(_ * 1.0/n)
    val sampleCovariance = covmat.matrixCovariance(m)

    (sampleMean, sampleCovariance)
  }

  /**
    * Calculate the mean and variance of an observation at tim t given a particle 
    * cloud representing the latent-state at time t and the model specification
    * @param mod a DGLM specification
    * @param xt the particle cloud representing the latent state at time t
    * @param v the observation noise variance
    * @return mean and variance of observation
    */
  def meanVarObservation(
    mod: Model,
    xt:  Vector[DenseVector[Double]],
    v:   DenseMatrix[Double]) = {

    for {
      ys <- xt traverse (x => mod.observation(x, v))
      (ft, qt) = meanCovSamples(ys)
    } yield (ft, qt)
  }

  /**
    * Forecast a DLM from a particle cloud representing the latent state
    * at the end of the observations
    * @param mod the model
    * @param xt the particle cloud representing the latent state
    * @param time the initial time to start the forecast from
    * @param p the parameters of the model
    * @return the time, mean observation and variance of the observation
    */
  def forecast(
    mod:  Model, 
    xt:   Vector[DenseVector[Double]], 
    time: Time,
    p:    Dlm.Parameters) = {

    MarkovChain((time, xt)){ case (t, x) => 
      for {
        x1 <- ParticleFilter.advanceState(mod.g, t + 1, x, p)
      } yield (t + 1, x1)
    }.steps.
      map { case (t, x) => (t, meanVarObservation(mod, x, p.v)) }
  }
}
