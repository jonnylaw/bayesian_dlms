package dlm

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.mean
import breeze.stats.distributions._
import cats._
import dlm.model.Dlm._

package object model {
  type State = MultivariateGaussian
  type Observation = DenseVector[Double]
  type Time = Int
  type ObservationMatrix = Time => DenseMatrix[Double]
  type SystemMatrix = Time => DenseMatrix[Double]

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
