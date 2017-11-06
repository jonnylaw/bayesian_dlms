package dlm

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.mean
import breeze.stats.distributions._
import cats._
import dlm.model.Dlm._

package object model {
  type Observation = DenseVector[Double]
  type Time = Double
  type ObservationMatrix = Time => DenseMatrix[Double]
  type SystemMatrix = Time => DenseMatrix[Double]
  type ConditionalLl = (Observation, DenseVector[Double]) => Double
  type LatentState = List[(Time, DenseVector[Double])]

  /**
    * A single observation of a model
    */
  case class Data(time: Time, observation: Option[Observation])

  /**
    * A Gaussian DLM can be implicitly converted to a DGLM
    * Then particle filtering methods can be used on Gaussian Models
    */
  implicit def dlm2dglm(dlmModel: Dlm.Model): Dglm.Model = {
    Dglm.Model(
      (x: DenseVector[Double], v: DenseMatrix[Double]) => MultivariateGaussianSvd(x, v),
      dlmModel.f,
      dlmModel.g,
      (p: Dlm.Parameters) => (x: DenseVector[Double], y: DenseVector[Double]) => MultivariateGaussianSvd(x, p.v).logPdf(y)
    )
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
