package core.dlm.model

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions._
import scala.math.{sin, cos}
import cats.implicits._

object Dlm {
  /**
    * A single observation of a model
    */
  case class Data(time: Double, observation: DenseVector[Option[Double]])

  /**
    * Parameters of a DLM
    */
  case class Parameters(
    v:  DenseMatrix[Double], 
    w:  DenseMatrix[Double], 
    m0: DenseVector[Double],
    c0: DenseMatrix[Double]
  ) {
    def map(f: Double => Double) = 
      Parameters(v.map(f), w.map(f), m0.map(f), c0.map(f))
  }

  /**
    * Definition of a DLM
    */
  case class Model(
    f: Double => DenseMatrix[Double], 
    g: Double => DenseMatrix[Double]) { self =>
    /**
      * Combine two DLMs into a multivariate DLM
      * @param y another DLM model
      * @return a DLM model
      */
    def |*|(y: Model): Model = {
      Dlm.outerSumModel(self, y)
    }

    def |+|(y: Model): Model = {
      Dlm.composeModels(self, y)
    }
  }

  /**
    * Dynamic Linear Models can be combined in order to model different
    * time dependent phenomena, for instance seasonal with trend
    */
    def composeModels(x: Model, y: Model): Model = {
      Model(
        (t: Double) => DenseMatrix.vertcat(x.f(t), y.f(t)), 
        (t: Double) => blockDiagonal(x.g(t), y.g(t))
      )
    }

  /**
    * Similar Dynamic Linear Models can be combined in order to model
    * multiple similar times series in a vectorised way
    */
  def outerSumModel(x: Model, y: Model) = {
    Model(
      (t: Double) => blockDiagonal(x.f(t), y.f(t)),
      (t: Double) => blockDiagonal(x.g(t), y.g(t))
    )
  }

  def outerSumParameters(x: Parameters, y: Parameters): Parameters = {
    Parameters(
      v = blockDiagonal(x.v, y.v),
      w = blockDiagonal(x.w, y.w),
      m0 = DenseVector.vertcat(x.m0, y.m0),
      c0 = blockDiagonal(x.c0, y.c0)
    )
  }


  /**
    * A polynomial model
    */
  def polynomial(order: Int): Model = {
    Model(
      (t: Double) => {
        val elements = Array.fill(order)(0.0)
        elements(0) = 1.0
        new DenseMatrix(order, 1, elements)
      },
      (dt: Double) => DenseMatrix.tabulate(order, order){ 
        case (i, j) if (i == j) => 1.0
        case (i, j) if (i == (j - 1)) => 1.0
        case _ => 0.0
      }
    )
  }

  /**
    * A first order regression model with intercept
    * @param x an array of covariates 
    */
  def regression(x: Array[DenseVector[Double]]): Model = {

    Model(
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
  def autoregressive(phi: Double*): Model = {
    Model(
      (t: Double) => {
        val m = DenseMatrix.zeros[Double](phi.size, 1)
        m(0,0) = 1.0
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
  def blockDiagonal(
    a: DenseMatrix[Double],
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
  def seasonalG(
    period:    Int,
    harmonics: Int)(
    dt:        Double): DenseMatrix[Double] = {

    val matrices = (delta: Double) => (1 to harmonics).
      map(h => Dlm.rotationMatrix(h * angle(period)(delta)))

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
  def seasonal(period: Int, harmonics: Int): Model = {
    Model(
      (t: Double) => DenseMatrix.tabulate(harmonics * 2, 1){ 
        case (h, i) => if (h % 2 == 0) 1 else 0 },
      (dt: Double) => seasonalG(period, harmonics)(dt)
    )
  }

  def stepState(
    x: DenseVector[Double],
    w: DenseMatrix[Double],
    g: Double => DenseMatrix[Double],
    dt: Double) = {

    for {
      w <- MultivariateGaussianSvd(DenseVector.zeros[Double](w.cols), w * dt)
      x1 = g(dt) * x + w
    } yield x1
  }

  def observation(
    x: DenseVector[Double],
    v: DenseMatrix[Double],
    f: Double => DenseMatrix[Double],
    time: Double) = {

    for {
      v <- MultivariateGaussianSvd(DenseVector.zeros[Double](v.cols), v)
      y = f(time).t * x + v
    } yield y
  }

  /**
    * Simulate a single step from a DLM, used in simulateRegular
    * @param mod a DLM model
    * @param x a realisation from the latent state at time t-1
    * @param time the current time
    * @param p the parameters of the DLM model
    * @param dt the time increment between successive realisations of the process
    */
  def simStep(
    mod:  Model, 
    x:    DenseVector[Double], 
    time: Double, 
    p:    Parameters,
    dt:   Double): Rand[(Data, DenseVector[Double])] = {
    for {
      x1 <- stepState(x, p.w, mod.g, dt)
      y <- observation(x, p.v, mod.g, time)
    } yield (Data(time, y.map(_.some)), x1)
  }

  /**
    * Simulate from a DLM
    */
  def simulateRegular(
    startTime: Double, 
    mod:       Model, 
    p:         Parameters,
    dt:        Double): Process[(Data, DenseVector[Double])] = {
  
    val initState = MultivariateGaussian(p.m0, p.c0).draw
    val init = (Data(startTime, DenseVector[Option[Double]](None)), initState)
    MarkovChain(init){ case (y, x) => simStep(mod, x, y.time + dt, p, dt) }
  }

  /**
    * Simulate the latent-state from a DLM model
    */
  def simulateStateRegular(
    mod: Model,
    w: DenseMatrix[Double]): Process[(Double, DenseVector[Double])] = {
    MarkovChain((1.0, DenseVector.zeros[Double](w.cols))){ case (time, x) => 
      MultivariateGaussianSvd(mod.g(1.0) * x, w).map((time + 1, _))
    }
  }

  /**
    * Simulate the state at the given times
    */
  def simulateState(
    times: Iterable[Double], 
    g:     Double => DenseMatrix[Double],
    p:     Dlm.Parameters,
    init:  (Double, DenseVector[Double])) = {

    times.tail.scanLeft(init) { (x, t) =>
      val dt = t - x._1
      (t, MultivariateGaussianSvd(g(dt) * x._2, p.w * dt).draw)
    }
  }

  /**
    * Simulate from a DLM at the given times
    */
  def simulate(times: Iterable[Double], mod: Model, p: Dlm.Parameters) = {
    val init = (times.head, MultivariateGaussianSvd(p.m0, p.c0).draw)
    val state = simulateState(times, mod.g, p, init)

    state.map { case (t, x) => 
      (Data(t, MultivariateGaussianSvd(mod.f(t).t * x, p.v).draw.map(_.some)), x) 
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
  def stepForecast(
    mod:  Model,
    time: Double,
    dt:   Double,
    mt:   DenseVector[Double], 
    ct:   DenseMatrix[Double],
    p:    Parameters) = {

    val (at, rt) = KalmanFilter.advanceState(mod.g, mt, ct, dt, p.w)
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
  def forecast(
    mod:  Model, 
    mt:   DenseVector[Double], 
    ct:   DenseMatrix[Double],
    time: Double,
    p:    Parameters) = {

    val (ft, qt) = KalmanFilter.oneStepPrediction(mod.f, mt, ct, time, p.v)

    Stream.iterate((time, mt, ct, ft, qt)){ 
      case (t, m, c, _, _) => stepForecast(mod, t + 1, 1.0, m, c, p) }.
      map(a => (a._1, a._4.data(0), a._5.data(0)))
  }

}
