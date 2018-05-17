package core.dlm

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.distributions.Rand
import cats._

package object model {
  /**
    * A Gaussian DLM can be implicitly converted to a DGLM
    * Then particle filtering methods can be used on Gaussian Models
    */
  implicit def dlm2dglm(dlmModel: Dlm.Model): Dglm.Model = {
    Dglm.Model(
      (x: DenseVector[Double], v: DenseMatrix[Double]) =>
        MultivariateGaussianSvd(x, v),
        dlmModel.f,
        dlmModel.g,
      (v: DenseMatrix[Double]) =>
      (x: DenseVector[Double], y: DenseVector[Double]) =>
        MultivariateGaussianSvd(x, v).logPdf(y)
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