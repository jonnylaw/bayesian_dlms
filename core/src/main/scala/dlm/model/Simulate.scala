package com.github.jonnylaw.dlm

import breeze.linalg.DenseVector
import breeze.stats.distributions._
import cats.implicits._

/**
  * Abstract trait to simulate data
  * @param M a model
  * @param P a set of parameters
  * @param S a single realisation of the latent-state
  */
trait Simulate[M, P, S] {
  def initialiseState(model: M, params: P): (Data, S)

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

  /**
    * Simulate from a model using regular steps
    */
  def simulateRegular(model: M, params: P, dt: Double): Process[(Data, S)] = {

    val init = initialiseState(model, params)
    MarkovChain(init) {
      case (y, x) => simStep(model, params)(x, y.time + dt, dt)
    }
  }
}
