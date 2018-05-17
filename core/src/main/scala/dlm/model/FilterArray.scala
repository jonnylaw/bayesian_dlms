package core.dlm.model

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.distributions._
import spire.syntax.cfor._
import scala.reflect.ClassTag

object FilterArray {
  /**
    * Perform the Kalman Filter using a cfor loop to be used in the Gibbs Sampler
    */
  def filter[S: ClassTag](
    ys:         Vector[Dlm.Data],
    step:       (S, Dlm.Data)  => S,
    initialise: S,
    st:         Array[S]
  ): Array[S] = {
    st(0) = initialise

    cfor(1)(_ < st.size, _ + 1) { i =>
      st(i) = step(st(i-1), ys(i-1))
    }

    st
  }

  def filterSvd(mod: Dlm.Model, ys: Vector[Dlm.Data], p: Dlm.Parameters) = {
    val sqrtVinv = SvdFilter.sqrtInvSvd(p.v)
    val sqrtW = SvdFilter.sqrtSvd(p.w)
    val params = p.copy(v = sqrtVinv, w = sqrtW)
    val st = Array.ofDim[SvdFilter.State](ys.length + 1)

    filter[SvdFilter.State](ys,
      SvdFilter.step(mod, params),
      SvdFilter.initialiseState(mod, params, ys), st)
  }

  def filterNaive(mod: Dlm.Model, ys: Vector[Dlm.Data], p: Dlm.Parameters) = {
    val st = Array.ofDim[KalmanFilter.State](ys.length + 1)

    filter[KalmanFilter.State](ys,
      KalmanFilter.step(mod, p),
      KalmanFilter.initialiseState(mod, p, ys), st)
  }

  /**
    * Perform backward sampling 
    */
  def sample[KS, S: ClassTag](
    filtered:   Array[KS],
    step:       (KS, S) => S,
    initialise: S,
    st:         Array[S]
  ) = {
    val n = filtered.size
    st(n-1) = initialise

    cfor(n - 2)(_ >= 0, _ - 1) { i =>
      st(i) = step(filtered(i), st(i+1))
    }

    st
  }

  def sampleSvd(
    mod:        Dlm.Model,
    w:          DenseMatrix[Double],
    filtered:   Array[SvdFilter.State]) = {

    val sqrtWinv = SvdFilter.sqrtInvSvd(w)
    val st = Array.ofDim[SvdSampler.State](filtered.size)

    sample[SvdFilter.State, SvdSampler.State](filtered,
      SvdSampler.step(mod, sqrtWinv), SvdSampler.initialise(filtered), st)
  }

  def sampleNaive(
    mod:        Dlm.Model,
    w:          DenseMatrix[Double],
    filtered:   Array[KalmanFilter.State]) = {

    val st = Array.ofDim[Smoothing.SamplingState](filtered.size)

    sample[KalmanFilter.State, Smoothing.SamplingState](filtered,
      Smoothing.step(mod, w), Smoothing.initialise(filtered.toVector), st)
  }

  /**
    * Forward filtering backward sampling by updating the array in place
    */
  def ffbsSvd(
    mod: Dlm.Model,
    ys:  Vector[Dlm.Data],
    p:   Dlm.Parameters) = {

    val filtered = filterSvd(mod, ys, p)
    Rand.always(sampleSvd(mod, p.w, filtered).map(a => (a.time, a.theta)))
  }

  def ffbsNaive(
    mod: Dlm.Model,
    ys:  Vector[Dlm.Data],
    p:   Dlm.Parameters): Rand[Array[(Double, DenseVector[Double])]] = {

    val filtered = filterNaive(mod, ys, p)
    Rand.always(sampleNaive(mod, p.w, filtered).map(a => (a.time, a.sample)))
  }

}
