package com.github.jonnylaw.dlm

import breeze.linalg.{DenseVector, DenseMatrix, diag, svd}
import breeze.stats.distributions.{Gaussian, Rand}
import cats.implicits._

/**
  * Backward Sampler utilising the SVD for stability
  */
object SvdSampler {

  /**
    * Perform a single step in the backward sampler using the SVD
    */
  def step(mod: Dlm, sqrtW: DenseMatrix[Double])(
      st: SvdState,
      ss: SamplingState): SamplingState = {

    val dt = ss.time - st.time
    val dcInv = st.dc.map(1.0 / _)
    val root = svd(DenseMatrix.vertcat(sqrtW * mod.g(dt) * st.uc, diag(dcInv)))

    val uh = st.uc * root.rightVectors.t
    val dh = root.singularValues.map(1.0 / _)

    val gWinv = mod.g(dt).t * sqrtW.t * sqrtW
    val du = diag(dh) * uh.t
    val h = st.mt + du.t * du * gWinv * (ss.sample - ss.at1)

    SamplingState(st.time,
                  rnorm(h, dh, uh).draw,
                  h,
                  uh * du,
                  st.at,
                  st.ur * diag(st.dr) * st.ur.t)
  }

  def initialise(filtered: Array[SvdState]) = {
    val last = filtered.last
    val sample = SvdSampler.rnorm(last.mt, last.dc, last.uc).draw
    val ct = last.uc * diag(last.dc) * last.uc.t
    val rt = last.ur * diag(last.dr) * last.ur.t

    SamplingState(last.time, sample, last.mt, ct, last.at, rt)
  }

  /**
    * Given a vector containing the SVD filtered results, perform backward sampling
    * @param mod a DLM specification
    * @param st the filtered state
    * @param w the square root of the system error matrix
    * @return
    */
  def sample(mod: Dlm,
             st: Vector[SvdState],
             sqrtW: DenseMatrix[Double]): Vector[SamplingState] = {

    val init = initialise(st.toArray)
    st.init.scanRight(init)(step(mod, sqrtW))
  }

  /**
    * Perform forward filtering backward sampling using
    * the SVD of the covariance matrix
    */
  def ffbs(mod: Dlm,
           ys: Vector[Data],
           p: DlmParameters,
           advState: (SvdState, Double) => SvdState) = {

    val ps = SvdFilter.transformParams(p)
    val filtered = SvdFilter(advState).filterDecomp(mod, ys, ps)
    Rand.always(sample(mod, filtered, ps.w))
  }

  /**
    * Perform FFBS for a DLM using the SVD
    */
  def ffbsDlm(mod: Dlm, ys: Vector[Data], p: DlmParameters) = {

    ffbs(mod, ys, p, SvdFilter.advanceState(p, mod.g))
  }

  /**
    * Simulate from a normal distribution given the right vectors and
    * singular values of the covariance matrix
    * @param mu the mean of the multivariate normal distribution
    * @param d the square root of the diagonal in the SVD of the
    * Error covariance matrix C_t
    * @param u the right vectors of the SVDfilter
    * @return a DenseVector sampled from the Multivariate Normal
    * distribution with mean mu and covariance u d^2 u^T
    */
  def rnorm(mu: DenseVector[Double],
            d: DenseVector[Double],
            u: DenseMatrix[Double]) = new Rand[DenseVector[Double]] {

    def draw = {
      val z = DenseVector.rand(mu.size, Gaussian(0, 1))
      mu + (u * diag(d) * z)
    }
  }

  def meanState(sampled: Seq[Seq[SamplingState]]) = {
    sampled.transpose
      .map(s =>
        (s.head.time, s.map(_.sample).reduce(_ + _) /:/ sampled.size.toDouble))
      .map { case (t, s) => List(t, s(0)) }
  }

  def intervalState(sampled: Seq[Seq[(Double, DenseVector[Double])]],
                    interval: Double = 0.95)
    : Seq[(Double, (DenseVector[Double], DenseVector[Double]))] = ???
}
