package core.dlm.model

import breeze.stats.distributions.MultivariateGaussian
import breeze.linalg.DenseMatrix

object FilterAr {
  /**
    * Advance the state mean and variance to the a-priori
    * value of the state at time t
    * @param st the kalman filter state
    * @param dt the time difference between observations (1.0)
    * @param p the parameters of the SV Model
    * @return the a-priori mean and covariance of the state at time t
    */
  def advanceState(
    p:  SvParameters)(
    st: KfState,
    dt: Double) = {

    st.copy(time = st.time + dt,
      at = st.mt map (m => p.mu + p.phi * (m - p.mu)),
      rt = st.ct map (c => p.phi * p.phi * c + p.sigmaEta * p.sigmaEta))
  }

  def backwardStep(
    p:        SvParameters,
  )(filtered: KfState,
    s:        Smoothing.SamplingState) = {

    // extract elements from kalman state
    val time = filtered.time
    val mt = filtered.mt
    val ct = filtered.ct
    val at1 = s.at1
    val rt1 = s.rt1

    val identity = DenseMatrix.eye[Double](mt.size)
    val g = identity * p.phi
    val w = identity * p.sigmaEta * p.sigmaEta
    val cgrinv = (rt1.t \ (g * ct.t)).t
    val mean = mt + cgrinv * (s.sample - at1)

    val diff = (identity - cgrinv * g)
    val cov = diff * ct * diff.t + cgrinv * w * cgrinv.t
    val sample = MultivariateGaussian(mean, cov).draw

    Smoothing.SamplingState(time, sample, filtered.at, filtered.rt)
  }

}
