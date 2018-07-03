package core.dlm.model

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.distributions._
import spire.syntax.cfor._
import scala.reflect.ClassTag
import cats.implicits._

object FilterArray {

  /**
    * Perform the Kalman Filter using a cfor loop to be used in the Gibbs Sampler
    */
  def filter[S: ClassTag](
      ys: Vector[Dlm.Data],
      step: (S, Dlm.Data) => S,
      initialise: S,
      st: Array[S]
  ): Array[S] = {
    st(0) = initialise

    cfor(1)(_ < st.size, _ + 1) { i =>
      st(i) = step(st(i - 1), ys(i - 1))
    }

    st
  }

  /**
    * Filter a DLM using the Singular value decomposition
    * @param mod the DLM specification
    * @param ys a time series of observations
    * @param p 
    */
  def filterSvd(mod: DlmModel, ys: Vector[Dlm.Data], p: DlmParameters) = {
    val sqrtVinv = SvdFilter.sqrtInvSvd(p.v)
    val sqrtW = SvdFilter.sqrtSvd(p.w)
    val params = p.copy(v = sqrtVinv, w = sqrtW)
    val st = Array.ofDim[SvdState](ys.length + 1)

    filter[SvdState](ys,
                     SvdFilter.step(mod, params, SvdFilter.advanceState(params, mod.g)),
                     SvdFilter.initialiseState(mod, params, ys),
                     st)
  }

  def filterNaive(mod: DlmModel, ys: Vector[Dlm.Data], p: DlmParameters) = {
    val st = Array.ofDim[KfState](ys.length + 1)

    filter[KfState](ys,
                    KalmanFilter.step(mod, p, KalmanFilter.advanceState(p, mod.g)),
                    KalmanFilter.initialiseState(mod, p, ys),
                    st)
  }

  /**
    * Perform backward sampling
    */
  def sample[KS, S: ClassTag](
      filtered: Array[KS],
      step: (KS, S) => S,
      initialise: S,
      st: Array[S]
  ) = {
    val n = filtered.size
    st(n - 1) = initialise

    cfor(n - 2)(_ >= 0, _ - 1) { i =>
      st(i) = step(filtered(i), st(i + 1))
    }

    st
  }

  def sampleSvd(mod: DlmModel,
                w: DenseMatrix[Double],
                filtered: Array[SvdState]) = {

    val sqrtWinv = SvdFilter.sqrtInvSvd(w)
    val st = Array.ofDim[SvdSampler.State](filtered.size)

    sample[SvdState, SvdSampler.State](filtered,
                                       SvdSampler.step(mod, sqrtWinv),
                                       SvdSampler.initialise(filtered),
                                       st)
  }

  def sampleNaive(mod: DlmModel,
                  w: DenseMatrix[Double],
                  filtered: Array[KfState]) = {

    val st = Array.ofDim[Smoothing.SamplingState](filtered.size)

    sample[KfState, Smoothing.SamplingState](
      filtered,
      Smoothing.step(mod, w),
      Smoothing.initialise(filtered.toVector),
      st)
  }

  /**
    * Forward filtering backward sampling by updating the array in place
    */
  def ffbsSvd(mod: DlmModel, ys: Vector[Dlm.Data], p: DlmParameters) = {

    val filtered = filterSvd(mod, ys, p)
    Rand.always(sampleSvd(mod, p.w, filtered).map(a => (a.time, a.theta)))
  }

  def ffbsNaive(
      mod: DlmModel,
      ys: Vector[Dlm.Data],
      p: DlmParameters): Rand[Array[(Double, DenseVector[Double])]] = {

    val filtered = filterNaive(mod, ys, p)
    Rand.always(sampleNaive(mod, p.w, filtered).map(a => (a.time, a.sample)))
  }
}
