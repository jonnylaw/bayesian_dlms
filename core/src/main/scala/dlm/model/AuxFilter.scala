package core.dlm.model

import breeze.stats.distributions._
import cats.Traverse
import cats.implicits._
import breeze.numerics.{log, exp}
import breeze.stats.mean

/**
  * Calculate an one-dimensional unknown observation variance 
  */
case class AuxFilter(n: Int) extends FilterTs[PfState, DlmParameters, DglmModel] {
  import ParticleFilter._

  /**
    * A single step of the Auxiliary Particle Filter
    */
  def step(
    mod: DglmModel,
    p:   DlmParameters)
    (s: PfState, d: Dlm.Data): PfState = {

    val y = KalmanFilter.flattenObs(d.observation)
    val dt = d.time - s.time

    if (y.data.isEmpty) {
      val x = advanceState(mod, dt, s.state, p).draw
      s.copy(state = x)
    } else {
      // draw from the prior (latent-state transition kernel)
      val newState = advanceState(mod, d.time, s.state, p).draw

      // calculate likelihood of new observation, conditional on newState
      val probs = (s.weights, newState).zipped.
        map { case (weight, state) => weight + calcWeight(mod, d.time, state, d.observation, p) }
      val max = probs.max
      val w1 = probs map { a => math.exp(a - max) }
      // select the particles to advance
      val particles = multinomialResample(s.state, w1)

      // advance new particles
      val x1 = advanceState(mod, dt, particles.toVector, p).draw

      // calculate weights
      val w = calcWeights(mod, d.time, x1, d.observation, p)
      val max1 = w.max
      val w2 = w map (a => exp(a - max1))
      val ll = s.ll + max + log(mean(w2))

      // resample again
      val resampledX = multinomialResample(x1, w2)

      PfState(d.time, resampledX, w2, ll)
    }
  }

  def initialiseState[T[_]: Traverse](
    model: DglmModel,
    p: DlmParameters,
    ys: T[Dlm.Data]): PfState = {

    val initState = MultivariateGaussian(p.m0, p.c0).sample(n).toVector
    val t0 = ys.map(_.time).reduceLeftOption((t0, d) => math.min(t0, d))
    PfState(t0.get - 1.0, initState, Vector.fill(n)(1.0 / n), 0.0)
  }
}
