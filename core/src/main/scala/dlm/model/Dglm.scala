package dlm.core.model

import breeze.stats.distributions._
import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.numerics.{lgamma, log}
import cats.implicits._
import math.exp
import breeze.stats.covmat

/**
  * A DGLM used for modelling non-linear
  * non-Gaussian univariate time series
  */
case class Dglm(
    observation: (DenseVector[Double],
                  DenseMatrix[Double]) => Rand[DenseVector[Double]],
    f: Double => DenseMatrix[Double],
    g: Double => DenseMatrix[Double],
    link: DenseVector[Double] => Double,
    conditionalLikelihood: (DenseMatrix[Double]) => (
        DenseVector[Double],
        DenseVector[Double]) => Double
)

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
    * Define a DGLM with Student's t observation errors
    */
  def studentT(nu: Int, mod: Dlm): Dglm = {
    Dglm(
      (x: DenseVector[Double], v: DenseMatrix[Double]) =>
        ScaledStudentsT(nu, x(0), math.sqrt(v(0, 0))).map(DenseVector(_)),
      mod.f,
      mod.g,
      x => x(0),
      conditionalLikelihood = (v: DenseMatrix[Double]) =>
        (x: DenseVector[Double], y: DenseVector[Double]) =>
          ScaledStudentsT(nu, x(0), math.sqrt(v(0, 0))).logPdf(y(0))
    )
  }

  def logit(p: Double): Double =
    log(p / (1 - p))

  def expit(x: Double): Double =
    exp(x) / (1 + exp(x))

  /**
    * Zero inflated Poisson DGLM, the observation variance is the logit of the
    * probability of observing a zero
    * @param mod the DLM model specifying the latent-state
    */
  def zip(mod: Dlm): Dglm = {
    Dglm(
      observation = (x: DenseVector[Double], v: DenseMatrix[Double]) => {
        val p = expit(v(0, 0))
        for {
          u <- Uniform(0, 1)
          nonZero <- Poisson(exp(x(0)))
          next = if (u < p) 0 else nonZero
        } yield DenseVector(next.toDouble)
      },
      mod.f,
      mod.g,
      x => exp(x(0)),
      conditionalLikelihood = (v: DenseMatrix[Double]) =>
        (x: DenseVector[Double], y: DenseVector[Double]) => {
          val obs: Double = y(0)
          val p = expit(v(0, 0))
          if (math.abs(obs - 0) < 1e-5) {
            log(p + (1.0 - p) * exp(-exp(x(0))))
          } else {
            -log(1.0 + exp(v(0, 0))) + obs * x(0) - exp(x(0)) - lgamma(
              obs + 1.0)
          }
      }
    )
  }

  /**
    * Negative Binomial Model for overdispersed count data
    */
  def negativeBinomial(mod: Dlm): Dglm = {
    Dglm(
      observation = (x: DenseVector[Double], logv: DenseMatrix[Double]) => {
        val size = exp(logv(0, 0))
        val prob = exp(x(0)) / (size + exp(x(0)))

        for {
          lambda <- Gamma(size, prob / (1 - prob))
          x <- Poisson(lambda)
        } yield DenseVector(x.toDouble)
      },
      mod.f,
      mod.g,
      x => exp(x(0)),
      conditionalLikelihood = (logv: DenseMatrix[Double]) =>
        (x: DenseVector[Double], y: DenseVector[Double]) => {
          val size = exp(logv(0, 0))
          val mu = exp(x(0))
          val obs: Double = y(0)

          lgamma(size + obs) - lgamma(obs + 1) - lgamma(size) +
            size * log(size / (mu + size)) + obs * log(mu / (mu + size))
      }
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
  def beta(mod: Dlm): Dglm = {

    Dglm(
      observation = (x, v) => {
        val mean = logisticFunction(1.0)(x(0))

        beta(mean, v(0, 0)).map(DenseVector(_))
      },
      f = mod.f,
      g = mod.g,
      x => logisticFunction(1.0)(x(0)),
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
  def poisson(mod: Dlm): Dglm = {
    Dglm(
      observation = (x, v) => Poisson(exp(x(0))).map(DenseVector(_)),
      f = mod.f,
      g = mod.g,
      x => exp(x(0)),
      conditionalLikelihood =
        v => (x, y) => Poisson(exp(x(0))).logProbabilityOf(y(0).toInt)
    )
  }

  def diagonal(m: DenseMatrix[Double]): Vector[Double] = {
    val res = for {
      i <- 0 until m.cols
    } yield m(i, i)

    res.toVector
  }

  /**
    * Advance the a multivariate state independently according to the
    * ornstein uhlenbeck process
    */
  def stepOu(model: Dglm, params: DlmParameters)(
      state: DenseVector[Double],
      dt: Double): Rand[DenseVector[Double]] = {

    val phi = model.g(dt)
    val variance = diagonal(params.w).zip(diagonal(phi)).map {
      case (w, ph) => (math.pow(w, 2) * (1 - exp(-2 * ph * dt))) / (2 * ph)
    }
    val mean =
      state.data.zip(diagonal(phi)).map { case (x, ph) => exp(-ph * dt) * x }

    val res = (mean zip variance).map { case (m, v) => Gaussian(m, v).draw }

    Rand.always(DenseVector(res))
  }

  def stepState(model: Dglm, params: DlmParameters)(
      state: DenseVector[Double],
      dt: Double): Rand[DenseVector[Double]] = {

    for {
      w <- MultivariateGaussianSvd(DenseVector.zeros[Double](params.w.cols),
                                   params.w * dt)
      x1 = model.g(dt) * state + w
    } yield x1
  }

  def simStep(model: Dglm, params: DlmParameters)(
      state: DenseVector[Double],
      time: Double,
      dt: Double): Rand[(Data, DenseVector[Double])] =
    for {
      x1 <- stepState(model, params)(state, dt)
      y <- observation(model, params, x1, time)
    } yield (Data(time, y.map(_.some)), x1)

  /**
    * Simulate from a model using regular steps
    */
  def simulateRegular(model: Dglm,
                      params: DlmParameters,
                      dt: Double): Process[(Data, DenseVector[Double])] = {

    val init = initialiseState(model, params)
    MarkovChain(init) {
      case (y, x) => simStep(model, params)(x, y.time + dt, dt)
    }
  }

  def observation(model: Dglm,
                  params: DlmParameters,
                  state: DenseVector[Double],
                  time: Double): Rand[DenseVector[Double]] = {

    model.observation(model.f(1.0).t * state, params.v)
  }

  def initialiseState(model: Dglm,
                      params: DlmParameters): (Data, DenseVector[Double]) = {

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

  def meanAndIntervals(prob: Double)(samples: Seq[DenseVector[Double]]) = {
    val n = samples.size
    val sampleMean = samples.reduce(_ + _).map(_ * 1.0 / n)
    val tsp = samples.map(_.data).toArray.transpose.map(_.sorted)
    val index = math.floor(n * prob).toInt
    val upper = tsp.map(v => v(index))
    val lower = tsp.map(v => v(n - index))

    (sampleMean, lower, upper)
  }

  def medianAndIntervals(prob: Double)(samples: Seq[DenseVector[Double]]) = {
    val n = samples.size
    val n2 = math.floor(n * 0.5).toInt
    val tsp = samples.map(_.data).toArray.transpose.map(_.sorted)
    val median = tsp.map(x => x(n2))

    val index = math.floor(n * prob).toInt
    val upper = tsp.map(v => v(index))
    val lower = tsp.map(v => v(n - index))

    (median, lower, upper)
  }

  /**
    * Forecast a DGLM from a particle cloud
    * representing the latent state
    * at the end of the observations
    * @param mod the model
    * @param xt the particle cloud representing the latent state
    * @param time the initial time to start the forecast from
    * @param p the parameters of the model
    * @return the time, mean observation and variance of the observation
    */
  def forecastParticles(mod: Dglm,
                        xt: Vector[DenseVector[Double]],
                        p: DlmParameters,
                        ys: Vector[Data]) = {

    val init = (ys.head.time,
                xt,
                Vector.fill(xt.size)(
                  DenseVector.zeros[Double](ys.head.observation.size)))

    ys.scanLeft(init) {
      case ((t0, x, _), y) =>
        val ys = y.observation.data.flatten
        val dt = y.time - t0
        val x1 = if (dt == 0) {
          x
        } else {
          ParticleFilter.advanceState(dt, x, mod, p).draw
        }
        val eta = x1 map (x => mod.f(y.time).t * x)
        val meanForecast = eta.map(e => mod.observation(e, p.v).draw)

        if (ys.isEmpty) {
          (y.time, x1, meanForecast)
        } else {
          val w = ParticleFilter.calcWeights(mod, y.time, x1, y.observation, p)
          val max = w.max
          val w1 = w map (a => exp(a - max))

          val resampled = ParticleFilter.multinomialResample(x1, w1)

          (y.time, resampled, meanForecast)
        }
    }
  }
}
