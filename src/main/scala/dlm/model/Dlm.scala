package dlm.model

import breeze.linalg.{DenseMatrix, diag, DenseVector, inv}
import breeze.stats.distributions._
import scala.math.{exp, log, sin, cos}
import cats.{Monad, Semigroup}
import cats.implicits._
import math.sqrt

object Dlm {
  /**
    * Definition of a DLM
    */
  case class Model(f: ObservationMatrix, g: SystemMatrix) { self =>
    /**
      * Combine two DLMs into a multivariate DLM
      * @param y another DLM model
      * @return a DLM model
      */
    def |*|(y: Model): Model = {
      Dlm.outerSumModel(self, y)
    }
  }

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
    * A polynomial model
    */
  def polynomial(order: Int): Model = {
    Model(
      (t: Time) => {
        val elements = Array.fill(order)(0.0)
        elements(0) = 1.0
        new DenseMatrix(order, 1, elements)
      },
      (dt: TimeIncrement) => DenseMatrix.tabulate(order, order){ 
        case (i, j) if (i == j) => 1.0
        case (i, j) if (i == (j - 1)) => 1.0
        case _ => 0.0
      }
    )
  }

  /**
    * A first order regression model with intercept
    */
  def regression(x: Array[DenseVector[Double]]): Model = {

    Model(
      (t: Time) => {
          val m = 1 + x(t.toInt).size
          new DenseMatrix(m, 1, 1.0 +: x(t.toInt).data)
      },
      (t: Time) => DenseMatrix.eye[Double](2)
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
    * TODO: Test and check this function
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

  def buildSeasonalMatrix(period: Int, harmonics: Int): DenseMatrix[Double] = {
    val freq = 2 * math.Pi / period
    val matrices = (1 to harmonics) map (h => rotationMatrix(freq * h))
    matrices.reduce(blockDiagonal)
  }

  def seasonalG(
    period:    Int,
    harmonics: Int)(
    dt:        TimeIncrement): DenseMatrix[Double] = {

    val matrices = (delta: TimeIncrement) => (1 to harmonics).
      map(h => Dlm.rotationMatrix(h * angle(period)(delta)))

    matrices(dt).reduce(Dlm.blockDiagonal)
  }

  def angle(period: Int)(dt: TimeIncrement): Double = {
    2 * math.Pi * (dt % period) / period
  }

  /**
    * Create a seasonal model with fourier components in the system evolution matrix
    * @param period the period of the seasonality
    * @param 
    */
  def seasonal(period: Int, harmonics: Int): Model = {
    Model(
      (t: Time) => DenseMatrix.tabulate(harmonics * 2, 1){ case (h, i) => if (h % 2 == 0) 1 else 0 },
      (dt: TimeIncrement) => seasonalG(period, harmonics)(dt)
    )
  }

  /**
    * Simulate a single step from a DLM
    */
  def simStep(
    mod: Model, 
    x: DenseVector[Double], 
    time: Time, 
    p: Parameters): Rand[(Data, DenseVector[Double])] = {

    for {
      w <- MultivariateGaussianSvd(DenseVector.zeros[Double](p.w.cols), p.w)
      v <- MultivariateGaussianSvd(DenseVector.zeros[Double](p.v.cols), p.v)
      x1 = mod.g(time) * x + w
      y = mod.f(time).t * x1 + v
    } yield (Data(time, Some(y)), x1)
  }

  /**
    * Simulate from a DLM
    */
  def simulateRegular(
    startTime: Time, 
    mod: Model, 
    p: Parameters): Process[(Data, DenseVector[Double])] = {

    val init = (Data(startTime, None), p.m0)
    MarkovChain(init){ case (y, x) => simStep(mod, x, y.time + 1, p) }
  }

  /**
    * Simulate the latent-state from a DLM model
    */
  def simulateStateRegular(
    mod: Model,
    w: DenseMatrix[Double]): Process[(Time, DenseVector[Double])] = {
    MarkovChain((1.0, DenseVector.zeros[Double](w.cols))){ case (time, x) => 
      MultivariateGaussianSvd(mod.g(time + 1) * x, w).map((time + 1, _))
    }
  }

  /**
    * Simulate the state at the given times
    */
  def simulateState(
    times: Iterable[Double], 
    g:     TimeIncrement => DenseMatrix[Double],
    p:     Dlm.Parameters,
    init:  (Time, DenseVector[Double])) = {

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
      (Data(t, Some(MultivariateGaussianSvd(mod.f(t).t * x, p.v).draw)), x) 
    }
  }

  /**
    * Dynamic Linear Models can be combined in order to model different
    * time dependent phenomena, for instance seasonal with trend
    */
  implicit def addModel = new Semigroup[Model] {
    def combine(x: Model, y: Model): Model = {
      Model(
        (t: Time) => DenseMatrix.vertcat(x.f(t), y.f(t)), 
        (t: Time) => blockDiagonal(x.g(t), y.g(t))
      )
    }
  }

  /**
    * Similar Dynamic Linear Models can be combined in order to model
    * multiple similar times series in a vectorised way
    */
  def outerSumModel(x: Model, y: Model) = {
    Model(
      (t: Time) => blockDiagonal(x.f(t), y.f(t)),
      (t: Time) => blockDiagonal(x.g(t), y.g(t))
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
    time: Time,
    mt:   DenseVector[Double], 
    ct:   DenseMatrix[Double],
    p:    Parameters) = {

    val (at, rt) = KalmanFilter.advanceState(mod.g, mt, ct, time, p.w)
    val (ft, qt) = KalmanFilter.oneStepPrediction(mod.f, at, rt, time, p.v)

    (time, at, rt, ft, qt)
  }

  /**
    * Forecast a DLM from a state
    */
  def forecast(
    mod:  Model, 
    mt:   DenseVector[Double], 
    ct:   DenseMatrix[Double],
    time: Time,
    p:    Parameters) = {

    val (ft, qt) = KalmanFilter.oneStepPrediction(mod.f, mt, ct, time, p.v)

    Stream.iterate((time, mt, ct, ft, qt)){ 
      case (t, m, c, _, _) => stepForecast(mod, t + 1, m, c, p) }.
      map(a => (a._1, a._4.data(0), a._5.data(0)))
  }

}
