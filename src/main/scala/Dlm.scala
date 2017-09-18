import breeze.linalg.{DenseMatrix, diag, DenseVector, inv}
import breeze.stats.distributions._
import scala.math.{exp, log, sin, cos}
import cats.{Monad, Semigroup}
import cats.implicits._

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
    log_v: Vector[Double], 
    log_w: Vector[Double], 
    m0: Vector[Double],
    log_c0: Vector[Double]
  ) {
    override def toString = (log_v ++ log_w ++ m0 ++ log_c0).mkString(", ")

    def traverse[G[_]: Monad](f: Double => G[Double]): G[Parameters] = {
      for {
        v <- log_v traverse f
        w <- log_w traverse f
        m <- m0 traverse f
        c0 <- log_c0 traverse f
      } yield Parameters(v, w, m, c0)
    }
  }

  /**
    * A single observation of a DLM
    */
  case class Data(time: Time, observation: Option[Observation])

  /**
    * A first order regression model with intercept
    */
  def regression(x: Array[DenseVector[Double]]): Model = {

    Model(
      (t: Time) => {
        val m = 1 + x(t).size
        new DenseMatrix(1, m, 1.0 +: x(t).data)
      },
      (t: Time) => DenseMatrix.eye[Double](x.head.size)
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
  def blockDiagonal(a: DenseMatrix[Double], 
    b: DenseMatrix[Double]): DenseMatrix[Double] = {
    val n = a.rows + b.rows
    val m = a.cols + b.cols

    DenseMatrix.tabulate(n, m){ case (i, j) => 
      if (i < a.rows && j < a.cols) {
        a(i, j)
      } else if (i >= a.rows && j >= a.cols) {
        b(i % a.rows, j % a.cols)
      } else {
        0.0
      }
    }
  }

  /**
    * Create a seasonal model with fourier components in the system evolution matrix
    */
  def seasonal(period: Int, harmonics: Int): Model = {
    val freq = 2 * math.Pi / period
    val matrices = (1 to harmonics) map (h => rotationMatrix(freq * h))

    Model(
      (t: Time) => DenseMatrix.tabulate(1, harmonics * 2){ case (i, h) => h % 2 },
      (t: Time) => matrices.reduce(blockDiagonal)
    )
  }

  /**
    * Dynamic Linear Models can be combined in order to model different
    * time dependent phenomena, for instance seasonal with trend
    */
  implicit val addModel = new Semigroup[Model] {
    def combine(x: Model, y: Model): Model = {
      def f(t: Time) = DenseMatrix.vertcat(x.f(t), y.f(t))
      def g(t: Time) = blockDiagonal(x.g(t), y.g(t))
      Model(f, g)
    }
  }

  def simStep(mod: Model, x: DenseVector[Double], time: Time, p: Parameters): Rand[(Data, DenseVector[Double])] = {
    for {
      w <- p.log_w.traverse(x => Gaussian(0.0, exp(x)): Rand[Double])
      v <- p.log_v.traverse(x => Gaussian(0.0, exp(x)): Rand[Double])
      x1 = mod.f(time).t * x + DenseVector(w.toArray)
      y = mod.g(time) * x1 + DenseVector(v.toArray)
    } yield (Data(time, Some(y)), x1)
  }

  /**
    * Simulate from a DLM
    */
  def simulate(startTime: Time, mod: Model, p: Parameters): Process[(Data, DenseVector[Double])] = {

    MarkovChain((Data(startTime, None), DenseVector(p.m0.toArray))){ case (y, x) => simStep(mod, x, y.time + 1, p) }
  }

  /**
    * Similar Dynamic Linear Models can be combined in order to model
    * multiple similar times series in a vectorised way
    */
  implicit val outerSumModel = new Semigroup[Model] {
    def combine(x: Model, y: Model): Model = ???
  }

  /**
    * Calculate the state given all the observed data
    */
  def backwardsSmoother(mod: Model, observations: Array[Data], p: Parameters) = ???

  implicit val randMonad = new Monad[Rand] {
    def pure[A](x: A): Rand[A] = Rand.always(x)
    def flatMap[A, B](fa: Rand[A])(f: A => Rand[B]): Rand[B] = 
      fa flatMap f

    def tailRecM[A, B](a: A)(f: A => Rand[Either[A, B]]): Rand[B] = f(a).draw match {
      case Left(a1) => tailRecM(a1)(f)
      case Right(b) => Rand.always(b)
    }
  }
}
