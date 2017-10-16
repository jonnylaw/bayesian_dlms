package dlm.model

import breeze.linalg.{DenseMatrix, diag, DenseVector, inv}
import breeze.stats.distributions._
import scala.math.{exp, log, sin, cos}
import cats.{Monad, Semigroup}
import cats.implicits._
import math.sqrt

object Dlm {
  type State = MultivariateGaussian
  type Observation = DenseVector[Double]
  type Time = Int
  type ObservationMatrix = Time => DenseMatrix[Double]
  type SystemMatrix = Time => DenseMatrix[Double]

  /**
    * Definition of a DLM
    */
  case class Model(f: ObservationMatrix, g: SystemMatrix)

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
    * A single observation of a DLM
    */
  case class Data(time: Time, observation: Option[Observation])

  def polynomial(order: Int): Model = {
    Model(
      (t: Time) => {
        val elements = Array.fill(order)(0.0)
        elements(0) = 1.0
        new DenseMatrix(order, 1, elements)
      },
      (t: Time) => DenseMatrix.tabulate(order, order){ 
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
          val m = 1 + x(t).size
          new DenseMatrix(m, 1, 1.0 +: x(t).data)
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

  /**
    * Create a seasonal model with fourier components in the system evolution matrix
    */
  def seasonal(period: Int, harmonics: Int): Model = {
    val freq = 2 * math.Pi / period
    val matrices = (1 to harmonics) map (h => rotationMatrix(freq * h))

    Model(
      (t: Time) => DenseMatrix.tabulate(harmonics * 2, 1){ case (h, i) => if (h % 2 == 0) 1 else 0 },
      (t: Time) => matrices.reduce(blockDiagonal)
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
      w <- MultivariateGaussian(DenseVector.zeros[Double](p.w.cols), p.w)
      v <- MultivariateGaussian(DenseVector.zeros[Double](p.v.cols), p.v)
      x1 = mod.g(time) * x + w
      y = mod.f(time).t * x1 + v
    } yield (Data(time, Some(y)), x1)
  }

  /**
    * Simulate from a DLM
    */
  def simulate(
    startTime: Time, 
    mod: Model, 
    p: Parameters): Process[(Data, DenseVector[Double])] = {

    val init = (Data(startTime, None), p.m0)
    MarkovChain(init){ case (y, x) => simStep(mod, x, y.time + 1, p) }
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
}
