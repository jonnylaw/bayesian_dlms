package com.github.jonnylaw.dlm

import cats.Traverse
import cats.implicits._
import spire.syntax.cfor._
import scala.reflect.ClassTag

/**
  * Abstract trait for a filter which allows the filter
  * to be performed on any collection which implementes Traverse
  */
trait FilterTs[S, P, M] {

  /**
    * Initialise the state of a filter
    */
  def initialiseState[T[_]: Traverse](model: M, p: P, ys: T[Data]): S

  /**
    * Perform a single step in the filter, parameterised by advance state
    */
  def step(model: M, p: P)(s: S, yt: Data): S

  /**
    * Perform the Filter on a traversable collection
    * this discards the initial state
    * @param model a time series model
    * @param ys a collection containing the observations of the process which implements
    * traverse
    * @param p the parameters of the model
    */
  def filterTraverse[T[_]: Traverse](model: M, ys: T[Data], p: P): T[S] = {

    val init = initialiseState(model, p, ys)
    FilterTs.scanLeft(ys, init, step(model, p))
  }

  /**
    * Filter using a vector
    */
  def filter(model: M, ys: Vector[Data], p: P): Vector[S] = {

    val init = initialiseState(model, p, ys)
    ys.scanLeft(init)(step(model, p))
  }

  /**
    * Perform the Filter using a cfor loop to be used in the Gibbs Sampler
    * @param
    */
  def filterArray(model: M, ys: Vector[Data], p: P)(
      implicit ct: ClassTag[S]): Array[S] = {

    val st = Array.ofDim[S](ys.length + 1)
    st(0) = initialiseState(model, p, ys)

    cfor(1)(_ < st.size, _ + 1) { i =>
      st(i) = step(model, p)(st(i - 1), ys(i - 1))
    }

    st.tail
  }
}

object FilterTs {

  /**
    * Traverse with state, like a scan left but for any traversable, does not include the initialial state
    */
  def scanLeft[T[_]: Traverse, A, B](xs: T[A],
                                     zero: B,
                                     f: (B, A) => B): T[B] = {
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
  def scanRight[T[_]: Traverse, A, B](xs: T[A], zero: B, f: (B, A) => B): T[B] =
    ???
}
