package core.dlm.model

import breeze.stats.distributions._
import breeze.linalg.{DenseVector, DenseMatrix}
import cats.implicits._
import math.exp
import breeze.stats.covmat
import Dlm.Data

/**
  * A DGLM
  */
case class DglmModel(
    observation: (DenseVector[Double],
                  DenseMatrix[Double]) => Rand[DenseVector[Double]],
    f: Double => DenseMatrix[Double],
    g: Double => DenseMatrix[Double],
    conditionalLikelihood: (DenseMatrix[Double]) => (
        DenseVector[Double],
        DenseVector[Double]) => Double
)

object Dglm extends Simulate[DglmModel, DlmParameters, DenseVector[Double]] {

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
    * Define a DGLM with Student's t observation errors
    */
  def studentT(df: Int, mod: DlmModel): DglmModel = {
    DglmModel(
      observation = (x: DenseVector[Double], v: DenseMatrix[Double]) =>
        ScaledStudentsT(df, x(0), v(0, 0)).map(DenseVector(_)),
      mod.f,
      mod.g,
      conditionalLikelihood = (v: DenseMatrix[Double]) =>
        (y: DenseVector[Double], x: DenseVector[Double]) =>
          ScaledStudentsT(df, x(0), v(0, 0)).logPdf(y(0))
    )
  }

  /**
    * A beta distribution parameterised by the mean and variance
    * @param mean the mean of the resulting beta distribution
    * @param variance the variance of the beta distribution
    * @return a beta distribution
    */
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
  def beta(mod: DlmModel): DglmModel = {

    DglmModel(
      observation = (x, v) => {
        val mean = logisticFunction(1.0)(x(0))

        beta(mean, v(0, 0)).map(DenseVector(_))
      },
      f = mod.f,
      g = mod.g,
      conditionalLikelihood = v =>
        (x, y) => {
          val mean = logisticFunction(1.0)(x(0))

          beta(mean, v(0, 0)).logPdf(y(0))
      }
    )
  }

  /**
    * Construct a DGLM with Poisson distributed observations
    */
  def poisson(mod: DlmModel): DglmModel = {
    DglmModel(
      observation = (x, v) => Poisson(exp(x(0))).map(DenseVector(_)),
      f = mod.f,
      g = mod.g,
      conditionalLikelihood =
        v => (x, y) => Poisson(exp(x(0))).logProbabilityOf(y(0).toInt)
    )
  }

  def stepState(
    model: DglmModel,
    params: DlmParameters,
    state: DenseVector[Double],
    dt: Double): Rand[DenseVector[Double]] = {

    for {
      w <- MultivariateGaussianSvd(
        DenseVector.zeros[Double](params.w.cols),
        params.w * dt)
      x1 = model.g(dt) * state + w
    } yield x1
  }

  def observation(
    model: DglmModel,
    params: DlmParameters,
    state: DenseVector[Double],
    time: Double): Rand[DenseVector[Double]] = {

    model.observation(state, params.v)
  }

  def initialiseState(
      model: DglmModel,
      params: DlmParameters): (Dlm.Data, DenseVector[Double]) = {

    val initState = MultivariateGaussianSvd(params.m0, params.c0).draw
    (Data(0, DenseVector[Option[Double]](None)), initState)
  }

  /**
    * Calculate the mean and covariance of a sequence of DenseVectors
    */
  def meanCovSamples(samples: Seq[DenseVector[Double]]) = {
    val n = samples.size
    val m = new DenseMatrix(n,
                            samples.head.size,
                            samples.map(_.data).toArray.transpose.flatten)
    val sampleMean = samples.reduce(_ + _).map(_ * 1.0 / n)
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
  def meanVarObservation(mod: DglmModel,
                         xt: Vector[DenseVector[Double]],
                         v: DenseMatrix[Double]) = {

    for {
      ys <- xt traverse (x => mod.observation(x, v))
      (ft, qt) = meanCovSamples(ys)
    } yield (ft, qt)
  }

  /**
    * Forecast a DGLM from a particle cloud representing the latent state
    * at the end of the observations
    * @param mod the model
    * @param xt the particle cloud representing the latent state
    * @param time the initial time to start the forecast from
    * @param p the parameters of the model
    * @return the time, mean observation and variance of the observation
    */
  def forecastParticles(mod: DglmModel,
                        xt: Vector[DenseVector[Double]],
                        time: Double,
                        p: DlmParameters) = {

    MarkovChain((time, xt)) {
      case (t, x) =>
        for {
          x1 <- ParticleFilter.advanceState(mod, 1.0, x, p)
        } yield (t + 1, x1)
    }.steps.map { case (t, x) => (t, meanVarObservation(mod, x, p.v).draw) }
  }
}
