package core.dlm.model

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions._
import scala.math.{sin, cos}
//import cats.implicits._

/**
  * Definition of a DLM
  */
case class DlmModel(f: Double => DenseMatrix[Double],
                    g: Double => DenseMatrix[Double]) { self =>

  /**
    * Combine two DLMs into a multivariate DLM
    * @param y another DLM model
    * @return a DLM model
    */
  def |*|(y: DlmModel): DlmModel =
    Dlm.outerSumModel(self, y)

  def |+|(y: DlmModel): DlmModel =
    Dlm.composeModels(self, y)
}

/**
  * Parameters of a DLM
  */
case class DlmParameters(
  v: DenseMatrix[Double],
  w: DenseMatrix[Double],
  m0: DenseVector[Double],
  c0: DenseMatrix[Double]) { self =>

  def map(f: Double => Double) =
    DlmParameters(v.map(f), w.map(f), m0.map(f), c0.map(f))

  def |*|(y: DlmParameters): DlmParameters =
    Dlm.outerSumParameters(self, y)
}

/**
  * A DLM with a p-vector of observations
  * y_t = F_t x_t + v_t, v_t ~ N(0, V)
  * x_t = F_t x_{t-1} + w_t, w_t ~ N(0, W)
  */
object Dlm extends Simulate[DlmModel, DlmParameters, DenseVector[Double]] {

  /**
    * A single observation of a model
    */
  case class Data(time: Double, observation: DenseVector[Option[Double]])

  /**
    * Dynamic Linear Models can be combined in order to model different
    * time dependent phenomena, for instance seasonal with trend
    */
  def composeModels(x: DlmModel, y: DlmModel): DlmModel = {
    DlmModel(
      (t: Double) => DenseMatrix.vertcat(x.f(t), y.f(t)),
      (t: Double) => blockDiagonal(x.g(t), y.g(t))
    )
  }

  /**
    * Similar Dynamic Linear Models can be combined in order to model
    * multiple similar times series in a vectorised way
    */
  def outerSumModel(x: DlmModel, y: DlmModel) = {
    DlmModel(
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
  def polynomial(order: Int): DlmModel = {
    DlmModel(
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
  def regression(x: Array[DenseVector[Double]]): DlmModel = {

    DlmModel(
      (t: Double) => {
        val m = 1 + x(t.toInt).size
        new DenseMatrix(m, 1, 1.0 +: x(t.toInt).data)
      },
      (dt: Double) => DenseMatrix.eye[Double](2)
    )
  }

  /**
    * Define a discrete time univariate first order autoregressive model
    * @param phi a sequence of autoregressive parameters of length equal
    * to the order of the autoregressive state
    */
  def autoregressive(phi: Double*): DlmModel = {
    DlmModel(
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
      (1 to harmonics).map(h => Dlm.rotationMatrix(h * angle(period)(delta)))

    matrices(dt).reduce(Dlm.blockDiagonal)
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
  def seasonal(period: Int, harmonics: Int): DlmModel = {
    DlmModel(
      (t: Double) =>
        DenseMatrix.tabulate(harmonics * 2, 1) {
          case (h, i) => if (h % 2 == 0) 1 else 0
      },
      (dt: Double) => seasonalG(period, harmonics)(dt)
    )
  }

  def stepState(model: DlmModel,
                p: DlmParameters,
                state: DenseVector[Double],
                dt: Double) = {

    for {
      w <- MultivariateGaussianSvd(DenseVector.zeros[Double](p.w.cols),
                                   p.w * dt)
      x1 = model.g(dt) * state + w
    } yield x1
  }

  def observation(model: DlmModel,
                  p: DlmParameters,
                  x: DenseVector[Double],
                  time: Double): Rand[DenseVector[Double]] = {

    for {
      v <- MultivariateGaussianSvd(DenseVector.zeros[Double](p.v.cols), p.v)
      y = model.f(time).t * x + v
    } yield y
  }

  def initialiseState(
      model: DlmModel,
      params: DlmParameters): (Dlm.Data, DenseVector[Double]) = {

    val x0 = MultivariateGaussianSvd(params.m0, params.c0).draw
    (Dlm.Data(1.0, DenseVector[Option[Double]](None)), x0)
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
  def stepForecast(
    mod: DlmModel,
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
    * @return a Stream of forecasts
    */
  def forecast(mod: DlmModel,
               mt: DenseVector[Double],
               ct: DenseMatrix[Double],
               time: Double,
               p: DlmParameters) = {

    val (ft, qt) = KalmanFilter.oneStepPrediction(mod.f, mt, ct, time, p.v)

    Stream
      .iterate((time, mt, ct, ft, qt)) {
        case (t, m, c, _, _) => stepForecast(mod, t + 1, 1.0, m, c, p)
      }
      .map(a => (a._1, a._4.data(0), a._5.data(0)))
  }
}
