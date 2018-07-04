package core.dlm.model

import breeze.linalg.DenseVector
import breeze.stats.distributions._
//import cats.Traverse
import cats.implicits._
import Dlm._

/**
  * Abstract trait to simulate data
  * @param M a model
  * @param P a set of parameters
  * @param S a single realisation of the latent-state
  */
trait Simulate[M, P, S] {
  def initialiseState(model: M, params: P): (Dlm.Data, S)

  def stepState(model: M, params: P, state: S, dt: Double): Rand[S]

  def observation(model: M,
                  params: P,
                  state: S,
                  time: Double): Rand[DenseVector[Double]]

  def simStep(model: M,
              params: P)(state: S, time: Double, dt: Double): Rand[(Data, S)] =
    for {
      x1 <- stepState(model, params, state, dt)
      y <- observation(model, params, x1, time)
    } yield (Data(time, y.map(_.some)), x1)

  // /**
  //   * Simulate from a model at the given times
  //   */
  // def simulate[T[_]: Traverse](times: T[Double],
  //                              model: M,
  //                              params: P): T[(Data, S)] = {
  //   val init = initialiseState(model, params)
  //   Filter.scan(times,
  //               init,
  //               p: (Double, (Dlm.Data, S)) =>
  //                 p match {
  //                   case (t: Double, x: (Dlm.Data, S)) =>
  //                     simStep(model, params)(x._2, t, t - x._1.time).draw
  //               })
  // }

  /**
    * Simulate from a model using regular steps
    */
  def simulateRegular(model: M, params: P, dt: Double): Process[(Data, S)] = {

    val init = initialiseState(model, params)
    MarkovChain(init) {
      case (y, x) => simStep(model, params)(x, y.time + dt, dt)
    }
  }

// def forecast(
}