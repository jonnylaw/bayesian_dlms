package core.dlm.model

import cats.Traverse
import cats.implicits._
import Dlm._

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
    Filter.scan(ys, init, step(model, p, advState))
  }
}

object Filter {

  /**
    * Traverse with state, like a scan but for any traversable
    */
  def scan[T[_]: Traverse, A, B](xs: T[A], zero: B, f: (B, A) => B): T[B] = {
    def run(a: A): cats.data.State[B, B] =
      for {
        prev <- cats.data.State.get[B]
        next = f(prev, a)
        _ <- cats.data.State.set(next)
      } yield next

    xs.traverse(run).runA(zero).value
  }
}
