package dlm.core.model

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.distributions.{Rand, MultivariateGaussian}
import cats.implicits._
import spire.syntax.cfor._
import scala.reflect.ClassTag
import math.exp

case class SamplingState(time: Double,
                         sample: DenseVector[Double],
                         mean: DenseVector[Double],
                         cov: DenseMatrix[Double],
                         at1: DenseVector[Double],
                         rt1: DenseMatrix[Double])

object Smoothing {
  case class SmoothingState(time: Double,
                            mean: DenseVector[Double],
                            covariance: DenseMatrix[Double],
                            at1: DenseVector[Double],
                            rt1: DenseMatrix[Double])

  /**
    * A single step in the backwards smoother
    * Requires that a Kalman Filter has been run on the model
    * @param mod a DLM model specification
    * @param state the state at time t + 1
    * @return
    */
  def smoothStep(mod: Dlm)(kfState: KfState, state: SmoothingState) = {

    // extract elements from kalman state
    val time = kfState.time
    val dt = state.time - kfState.time
    val mt = kfState.mt
    val ct = kfState.ct
    val at1 = state.at1
    val rt1 = state.rt1

    val cgrinv = (rt1.t \ (mod.g(dt) * ct.t)).t

    val mean = mt + cgrinv * (state.mean - at1)
    val covariance = ct - cgrinv * (rt1 - state.covariance) * cgrinv

    SmoothingState(time, mean, covariance, kfState.at, kfState.rt)
  }

  /**
    * Learn the distribution of the latent state (the smoothing distribution)
    * p(x_{1:T} | y_{1:T}) of a fully specified DLM with observations available
    * for all time
    * @param mod a DLM model specification
    * @param kfState the output of a Kalman Filter
    * @return
    */
  def backwardsSmoother(mod: Dlm)(kfState: Vector[KfState]) = {

    val last = kfState.last
    val lastTime = last.time
    val init = SmoothingState(lastTime, last.mt, last.ct, last.at, last.rt)

    kfState.init.scanRight(init)(smoothStep(mod))
  }

  /**
    * Perform a single step of the Backward Sampling algorithm for a DLM
    * @param mod a DLM
    * @param w the system noise covariance matrix
    * @param kfState the state at the Kalman Filter at time t
    * @param state the sampling state at time t + 1
    * @return a sample from the state
    */
  def step(mod: Dlm, w: DenseMatrix[Double])(kfState: KfState,
                                             state: SamplingState) = {

    // extract elements from kalman state
    val dt = state.time - kfState.time
    val mt = kfState.mt
    val ct = kfState.ct
    val at1 = state.at1
    val rt1 = state.rt1

    // more efficient than inverting rt, equivalent to C * G.t * inv(R)
    val cgrinv = (rt1.t \ (mod.g(dt) * ct.t)).t

    // calculate the updated mean
    val mean = mt + cgrinv * (state.sample - at1)

    // calculate the updated covariance
    val n = mt.size
    val identity = DenseMatrix.eye[Double](n)
    val diff = identity - cgrinv * mod.g(dt)
    val covariance = diff * ct * diff.t + cgrinv * w * dt * cgrinv.t
    val r = (covariance + covariance.t) /:/ 2.0

    SamplingState(kfState.time,
                  MultivariateGaussianSvd(mean, r).draw,
                  kfState.mt,
                  kfState.ct,
                  kfState.at,
                  kfState.rt)
  }

  def initialise(filtered: Vector[KfState]) = {
    val last = filtered.last
    val lastState = MultivariateGaussianSvd(last.mt, last.ct).draw
    SamplingState(last.time, lastState, last.mt, last.ct, last.at, last.rt)
  }

  /**
    * Perform backward sampling
    */
  def sample(mod: Dlm,
             filtered: Vector[KfState],
             backStep: (KfState, SamplingState) => SamplingState) = {

    val initState = initialise(filtered)

    filtered.init
      .scanRight(initState)(backStep)
  }

  /**
    * Perform backward sampling using a cfor loop
    */
  def sampleArray[FS, S, M](
      filtered: Array[FS],
      initialise: S,
      backStep: (FS, S) => S)(implicit ct: ClassTag[S]): Array[S] = {

    val n = filtered.size
    val st: Array[S] = Array.ofDim[S](filtered.size)
    st(n - 1) = initialise

    cfor(n - 2)(_ >= 0, _ - 1) { i =>
      st(i) = backStep(filtered(i), st(i + 1))
    }

    st
  }

  /**
    * Forward filtering backward sampling
    * @param mod the DLM
    * @param observations a list of observations
    * @param advState the a priori state update in a Kalman Filter
    * @param backStep the backward step of the backward sampler
    * @param p parametes of the DLM
    */
  def ffbs(mod: Dlm,
           observations: Vector[Data],
           advState: (KfState, Double) => KfState,
           backStep: (KfState, SamplingState) => SamplingState,
           p: DlmParameters) = {

    val filtered = KalmanFilter(advState).filter(mod, observations, p)
    Rand.always(sample(mod, filtered, backStep))
  }

  /**
    * Perform backward sampling for a DLM
    */
  def sampleDlm(mod: Dlm, filtered: Vector[KfState], w: DenseMatrix[Double]) =
    sample(mod, filtered, Smoothing.step(mod, w))

  /**
    * Forward filtering backward sampling for a DLM
    * @param mod the DLM
    * @param observations a list of observations
    * @param p parametes of the DLM
    */
  def ffbsDlm(mod: Dlm, ys: Vector[Data], p: DlmParameters) = {

    Smoothing.ffbs(mod,
                   ys,
                   KalmanFilter.advanceState(p, mod.g),
                   Smoothing.step(mod, p.w),
                   p)
  }

  /**
    * Perform a backward step of FFBS for the OU process
    */
  def backwardStepOu(p: SvParameters)(kfState: KfState,
                                      s: SamplingState): SamplingState = {

    val dt = s.time - kfState.time
    val phi = exp(-p.phi * dt)
    val variance = (math.pow(p.sigmaEta, 2) / (2 * p.phi)) * (1 - exp(
      -2 * p.phi * dt))

    // extract elements from kalman state
    val time = kfState.time
    val mt = kfState.mt
    val ct = kfState.ct
    val at1 = s.at1
    val rt1 = s.rt1

    val identity = DenseMatrix.eye[Double](mt.size)
    val g = identity * phi
    val w = identity * variance
    val cgrinv = (rt1.t \ (g * ct.t)).t
    val mean = mt + cgrinv * (s.sample - at1)

    val diff = (identity - cgrinv * g)
    val cov = diff * ct * diff.t + cgrinv * w * cgrinv.t
    val sample = MultivariateGaussian(mean, cov).draw

    SamplingState(time, sample, mean, cov, kfState.at, kfState.rt)
  }
}
