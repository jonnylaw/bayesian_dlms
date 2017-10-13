package dlm.model

import cats._
import cats.data._
import cats.implicits._
import breeze.linalg.DenseVector

trait TimeSeries[F[_]] extends Monad[F] with Traverse[F]

object TimeSeries {
  def interpolate(list: Array[Option[Double]]) = {
    val prevs = list.zipWithIndex.scanLeft(Option.empty[(Double, Int)]) {
      case (prev, (cur, i)) => cur.map((_, i)).orElse(prev)
    }
    val nexts = list.zipWithIndex.scanRight(Option.empty[(Double, Int)]) {
      case ((cur, i), next) => cur.map((_, i)).orElse(next)
    }
    prevs.tail.zip(nexts).zipWithIndex.map {
      case ((Some((prev, i)), Some((next, j))), k) =>
        if (i == j) prev else prev + (next - prev) * (k - i).toDouble / (j - i).toDouble
      case ((Some((prev, _)), _), _) => prev
      case ((_, Some((next, _))), _) => next
    }
  }
}
