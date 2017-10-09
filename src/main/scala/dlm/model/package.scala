package dlm

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.mean
import breeze.stats.distributions._
import cats.{Eq, Monad}
import dlm.model.Dlm._

package object model {
  type State = MultivariateGaussian
  type Observation = DenseVector[Double]
  type Time = Int
  type ObservationMatrix = Time => DenseMatrix[Double]
  type SystemMatrix = Time => DenseMatrix[Double]

  val tolerance = 0.5

  // implicit def doubleEq = new Eq[Double] {
  //   def eqv(x: Double, y: Double) = {
  //  
  //   }
  // }

  implicit def vectoreq = new Eq[DenseVector[Double]] {
    def eqv(x: DenseVector[Double], y: DenseVector[Double]) = {
      x.data.zip(y.data).
        forall { case (a, b) => math.abs(a - b) < tolerance }
    }
  }

  implicit def dataeq = new Eq[Data] {
    def eqv(x: Data, y: Data) = (x.observation, y.observation) match {
      case (Some(a), Some(b)) =>
        x.time == y.time & implicitly[Eq[DenseVector[Double]]].eqv(a, b)
      case (None, None) =>
        x.time == y.time
      case _ => false
    }
  }

  implicit def matrixeq = new Eq[DenseMatrix[Double]] {
    def eqv(x: DenseMatrix[Double], y: DenseMatrix[Double]) = {
      x.data.zip(y.data).
        forall { case (a, b) => math.abs(a - b) < 1 }
    }
  }

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
