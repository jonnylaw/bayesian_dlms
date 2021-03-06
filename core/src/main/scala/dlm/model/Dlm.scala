package com.github.jonnylaw.dlm

import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions._
import scala.math.{sin, cos}
import cats.Semigroup
import cats.implicits._

/**
  * A state space model with a linear Gaussian latent-state
  * @param f the observation matrix which can be a function of time
  * @param g the system matrix
  */
case class Dlm(f: Double => DenseMatrix[Double],
               g: Double => DenseMatrix[Double]) { self =>

  /**
    * Combine two DLMs into a multivariate DLM
    * @param y another DLM
    * @return a multivariate DLM
    */
  def |*|(y: Dlm): Dlm =
    Dlm.outerSumModel(self, y)

  /**
    * Combine two univariate DLMs into a rich univariate DLM
    * @param y another DLM
    */
  def |+|(y: Dlm): Dlm =
    Dlm.composeModels(self, y)
}

/**
  * Parameters of a DLM
  */
case class DlmParameters(v: DenseMatrix[Double],
                         w: DenseMatrix[Double],
                         m0: DenseVector[Double],
                         c0: DenseMatrix[Double]) { self =>

  def map(f: Double => Double) =
    DlmParameters(v.map(f), w.map(f), m0.map(f), c0.map(f))

  def add(p: DlmParameters) =
    p.copy(v = p.v + v, w = p.w + w, m0 = p.m0 + m0, c0 = p.c0 + c0)

  def minus(p: DlmParameters) =
    p.copy(v = p.v - v, w = p.w - w, m0 = p.m0 - m0, c0 = p.c0 - c0)

  def times(p: DlmParameters) =
    p.copy(v = p.v * v, w = p.w * w, m0 = p.m0 * m0, c0 = p.c0 * c0)

  def |*|(y: DlmParameters): DlmParameters =
    Dlm.outerSumParameters(self, y)

  def toList = DlmParameters.toList(self)
}

object DlmParameters {
  def apply(v: Double, w: Double, m0: Double, c0: Double): DlmParameters =
    DlmParameters(DenseMatrix(v),
                  DenseMatrix(w),
                  DenseVector(m0),
                  DenseMatrix(c0))

  def empty(vDim: Int, wDim: Int): DlmParameters =
    DlmParameters(
      v = DenseMatrix.zeros[Double](vDim, vDim),
      w = DenseMatrix.zeros[Double](wDim, wDim),
      m0 = DenseVector.zeros[Double](wDim),
      c0 = DenseMatrix.zeros[Double](wDim, wDim)
    )

  def fromList(vDim: Int, wDim: Int)(l: List[Double]) =
    DlmParameters(
      diag(DenseVector(l.take(vDim).toArray)),
      diag(DenseVector(l.slice(vDim, vDim + wDim).toArray)),
      DenseVector(l.slice(vDim + wDim, vDim + 2 * wDim).toArray),
      diag(DenseVector(l.slice(vDim + 2 * wDim, vDim + 3 * wDim).toArray))
    )

  def toList(p: DlmParameters): List[Double] =
    DenseVector.vertcat(diag(p.v), diag(p.w), p.m0, diag(p.c0)).data.toList

  implicit def dlmSemigroup = new Semigroup[DlmParameters] {
    def combine(x: DlmParameters, y: DlmParameters) =
      x add y
  }
}

/**
  * A single observation of a model
  */
case class Data(time: Double, observation: DenseVector[Option[Double]])

/**
  * A DLM with a p-vector of observations
  * y_t = F_t x_t + v_t, v_t ~ N(0, V)
  * x_t = F_t x_{t-1} + w_t, w_t ~ N(0, W)
  */
object Dlm {

  /**
    * Dynamic Linear Models can be combined in order to model different
    * time dependent phenomena, for instance seasonal with trend
    */
  def composeModels(x: Dlm, y: Dlm): Dlm =
    Dlm(
      (t: Double) => DenseMatrix.vertcat(x.f(t), y.f(t)),
      (dt: Double) => blockDiagonal(x.g(dt), y.g(dt))
    )

  /**
    * Similar Dynamic Linear Models can be combined in order to model
    * multiple similar times series in a vectorised way
    */
  def outerSumModel(x: Dlm, y: Dlm) = {
    Dlm(
      (t: Double) => blockDiagonal(x.f(t), y.f(t)),
      (t: Double) => blockDiagonal(x.g(t), y.g(t))
    )
  }

  /**
    * Combine parameters of univariate models appropriately for a multivariate model
    */
  def outerSumParameters(x: DlmParameters, y: DlmParameters): DlmParameters = {
    DlmParameters(
      v = blockDiagonal(x.v, y.v),
      w = blockDiagonal(x.w, y.w),
      m0 = DenseVector.vertcat(x.m0, y.m0),
      c0 = blockDiagonal(x.c0, y.c0)
    )
  }

  /**
    * A polynomial model
    */
  def polynomial(order: Int): Dlm = {
    Dlm(
      (t: Double) => {
        val elements = Array.fill(order)(0.0)
        elements(0) = 1.0
        new DenseMatrix(order, 1, elements)
      },
      (dt: Double) =>
        DenseMatrix.tabulate(order, order) {
          case (i, j) if (i == j)       => 1.0
          case (i, j) if (i == (j - 1)) => 1.0
          case _                        => 0.0
      }
    )
  }

  /**
    * A first order regression model with intercept
    * @param x an array of covariates
    */
  def regression(x: Array[DenseVector[Double]]): Dlm = {

    Dlm(
      (t: Double) => {
        val index = t.toInt - 1
        val m = 1 + x(index).size
        new DenseMatrix(m, 1, 1.0 +: x(index).data)
      },
      (dt: Double) => DenseMatrix.eye[Double](2)
    )
  }

  /**
    * Define a discrete time univariate autoregressive model
    * @param phi a sequence of autoregressive parameters of length equal
    * to the order of the autoregressive state
    */
  def autoregressive(phi: Double*): Dlm = {
    Dlm(
      (t: Double) => {
        val m = DenseMatrix.zeros[Double](phi.size, 1)
        m(0, 0) = 1.0
        m
      },
      (dt: Double) => new DenseMatrix(phi.size, 1, phi.toArray)
    )
  }

  /**
    * Build a 2 x 2 rotation matrix
    */
  def rotationMatrix(theta: Double): DenseMatrix[Double] = {
    DenseMatrix((cos(theta), -sin(theta)), (sin(theta), cos(theta)))
  }

  /**
    * Build a block diagonal matrix by combining two matrices of the same size
    */
  def blockDiagonal(a: DenseMatrix[Double],
                    b: DenseMatrix[Double]): DenseMatrix[Double] = {

    val right = DenseMatrix.zeros[Double](a.rows, b.cols)

    val left = DenseMatrix.zeros[Double](b.rows, a.cols)

    DenseMatrix.vertcat(
      DenseMatrix.horzcat(a, right),
      DenseMatrix.horzcat(left, b)
    )
  }

  /**
    * Build the G matrix for the system evolution
    */
  def seasonalG(period: Int, harmonics: Int)(
      dt: Double): DenseMatrix[Double] = {

    val matrices = (delta: Double) =>
      (1 to harmonics).map(h => rotationMatrix(h * angle(period)(delta)))

    matrices(dt).reduce(blockDiagonal)
  }

  /**
    * Get the angle of the rotation for the seasonal model
    */
  def angle(period: Int)(dt: Double): Double = {
    2 * math.Pi * (dt % period) / period
  }

  /**
    * Create a seasonal model with fourier components in the system evolution matrix
    * @param period the period of the seasonality
    * @param harmonics the number of harmonics in the seasonal model
    * @return a seasonal DLM model
    */
  def seasonal(period: Int, harmonics: Int): Dlm = {
    Dlm(
      (t: Double) =>
        DenseMatrix.tabulate(harmonics * 2, 1) {
          case (h, i) => if (h % 2 == 0) 1 else 0
      },
      (dt: Double) => seasonalG(period, harmonics)(dt)
    )
  }

  def stepState(model: Dlm,
                p: DlmParameters,
                state: DenseVector[Double],
                dt: Double) = {

    for {
      w <- MultivariateGaussianSvd(DenseVector.zeros[Double](p.w.cols),
                                   p.w * dt)
      x1 = model.g(dt) * state + w
    } yield x1
  }

  def observation(model: Dlm,
                  p: DlmParameters,
                  x: DenseVector[Double],
                  time: Double): Rand[DenseVector[Double]] = {

    for {
      v <- MultivariateGaussianSvd(DenseVector.zeros[Double](p.v.cols), p.v)
      y = model.f(time).t * x + v
    } yield y
  }

  def initialiseState(model: Dlm,
                      params: DlmParameters): (Data, DenseVector[Double]) = {

    val x0 = MultivariateGaussianSvd(params.m0, params.c0).draw
    (Data(0.0, DenseVector[Option[Double]](None)), x0)
  }

  def simStep(model: Dlm, params: DlmParameters)(
      state: DenseVector[Double],
      time: Double,
      dt: Double): Rand[(Data, DenseVector[Double])] =
    for {
      x1 <- stepState(model, params, state, dt)
      y <- observation(model, params, x1, time)
    } yield (Data(time, y.map(_.some)), x1)

  def simulateRegular(model: Dlm,
                      params: DlmParameters,
                      dt: Double): Process[(Data, DenseVector[Double])] = {

    val init = initialiseState(model, params)
    MarkovChain(init) {
      case (y, x) => simStep(model, params)(x, y.time + dt, dt)
    }
  }

  /**
    * Perform a single forecast step, equivalent to performing the Kalman Filter
    * Without an observation of the process
    * @param mod a DLM specification
    * @param time the current time
    * @param mt the mean of the latent state at time t
    * @param ct the variance of the latent state at time t
    * @param p the parameters of the DLM
    */
  def stepForecast(mod: Dlm,
                   time: Double,
                   dt: Double,
                   mt: DenseVector[Double],
                   ct: DenseMatrix[Double],
                   p: DlmParameters) = {

    val (at, rt) = KalmanFilter.advState(mod.g, mt, ct, dt, p.w)
    val (ft, qt) = KalmanFilter.oneStepPrediction(mod.f, at, rt, time, p.v)

    (time, at, rt, ft, qt)
  }

  /**
    * Forecast a DLM from a state
    * @param mod a DLM
    * @param mt the posterior mean of the state at time t (start of forecast)
    * @param ct the posterior variance of the state at time t (start of forecast)
    * @param time the starting time of the forecast
    * @param p the parameters of the DLM
    * @return a Stream containing the time, forecast mean and variance
    */
  def forecast(mod: Dlm,
               mt: DenseVector[Double],
               ct: DenseMatrix[Double],
               time: Double,
               p: DlmParameters) = {

    val (ft, qt) = KalmanFilter.oneStepPrediction(mod.f, mt, ct, time, p.v)

    Stream
      .iterate((time, mt, ct, ft, qt)) {
        case (t, m, c, _, _) => stepForecast(mod, t + 1, 1.0, m, c, p)
      }
      .map(a => (a._1, a._4, a._5))
  }

  /**
    * Summarise forecast
    */
  def summariseForecast(interval: Double)(
      ft: DenseVector[Double],
      qt: DenseMatrix[Double]): List[List[Double]] = {

    for {
      i <- List.range(0, ft.size)
      g = Gaussian(ft(i), qt(i, i))
      upper = g.inverseCdf(interval)
      lower = g.inverseCdf(1 - interval)
    } yield List(ft(i), lower, upper)

  }
}
