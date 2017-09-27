package dlm.model

import breeze.linalg.DenseVector
import breeze.stats.distributions._
import Dlm._
import cats.implicits._

/**
  * The particle filter can be used for inference of DGLMs
  * ie. Dynamic Generalised Linear Models, where the observation distribution is not Gaussian
  */
object ParticleFilter {
  type CondLikelihood = (Observation, DenseVector[Double]) => Double

  case class State(
    time: Time,
    state: Vector[DenseVector[Double]]
  )

  def advanceState(mod: Model, time: Time, state: Vector[DenseVector[Double]], p: Parameters) = {
    state traverse (x => MultivariateGaussian(mod.g(time) * x , p.w): Rand[DenseVector[Double]])
  }

  def calcWeights(
    mod: Model, 
    time: Time, 
    state: Vector[DenseVector[Double]], 
    p: Parameters, 
    y: Observation,
    condLl: CondLikelihood
  ) = {
    state.map(x => condLl(y, mod.f(time) * x))
  }

  /**
    * Multinomial Resampling
    */
  def resample[A](particles: Vector[A], weights: Vector[Double]): Vector[A] = {
    val indices = Vector.fill(particles.size)(Multinomial(DenseVector(weights.toArray)).draw)

    indices map (particles(_))
  }

  def filterStep(mod: Model, p: Parameters, 
    conditionalLikelihood: CondLikelihood)(state: State, d: Data): State = d.observation match {
    case Some(y) =>
      val x1 = advanceState(mod, d.time, state.state, p).draw
      val w = calcWeights(mod, d.time, x1, p, y, conditionalLikelihood)
      val x = resample(x1, w)

      State(d.time, x)
    case None =>
      val x = advanceState(mod, d.time, state.state, p).draw
      State(d.time, x)
  }

  def filter(mod: Model, observations: Array[Data], p: Parameters, n: Int, conditionalLikelihood: CondLikelihood) = {
    val initState = MultivariateGaussian(p.m0, p.c0).sample(n).toVector
    val init = State(observations.head.time, initState)

    observations.scanLeft(init)(filterStep(mod, p, conditionalLikelihood))
  }
}
