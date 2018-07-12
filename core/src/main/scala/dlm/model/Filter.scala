package core.dlm.model

import cats.Traverse
import cats.implicits._
import Dlm._
import spire.syntax.cfor._
import scala.reflect.ClassTag

/**
  * Abstract trait for a filter which allows the filter
  * to be performed on any collection which implementes Traverse
  */
trait Filter[S, P, M] {

  /**
    * Initialise the state of a filter
    */
  def initialiseState[T[_]: Traverse](model: M, p: P, ys: T[Data]): S

  /**
    * Perform a single step in the filter, parameterised by advance state
    */
  def step(model: M, p: P, advState: (S, Double) => S)(s: S, yt: Data): S

  /**
    * Perform the Filter on a traversable collection
    */
  def filter[T[_]: Traverse](model: M, ys: T[Data], p: P, advState: (S, Double) => S): T[S] = {

    val init = initialiseState(model, p, ys)
    Filter.scanLeft(ys, init, step(model, p, advState))
  }

  /**
    * Filter using a vector
    */
  def filterVector(model: M, ys: Vector[Data], p: P, advState: (S, Double) => S): Seq[S] = {

    val init = initialiseState(model, p, ys)
    ys.scanLeft(init)(step(model, p, advState))
  }

  /**
    * Perform the Filter using a cfor loop to be used in the Gibbs Sampler
    * @param 
    */
  def filterArray(
    model:    M,
    ys:       Vector[Dlm.Data],
    p:        P,
    advState: (S, Double) => S)(implicit ct: ClassTag[S]): Array[S] = {

    val st = Array.ofDim[S](ys.length + 1)
    st(0) = initialiseState(model, p, ys)

    cfor(1)(_ < st.size, _ + 1) { i =>
      st(i) = step(model, p, advState)(st(i - 1), ys(i - 1))
    }

    st.tail
  }
}

object Filter {

  /**
    * Traverse with state, like a scan left but for any traversable, does not include the initialial state
    */
  def scanLeft[T[_]: Traverse, A, B](xs: T[A], zero: B, f: (B, A) => B): T[B] = {
    def run(a: A): cats.data.State[B, B] =
      for {
        prev <- cats.data.State.get[B]
        next = f(prev, a)
        _ <- cats.data.State.set(next)
      } yield next

    xs.traverse(run).runA(zero).value
  }

  /**
    * https://tech-blog.capital-match.com/posts/5-the-reverse-state-monad.html
    */
  def scanRight[T[_]: Traverse, A, B](xs: T[A], zero: B, f: (B, A) => B): T[B] = ???
}
