package core.dlm.model

import math.exp
import breeze.linalg.DenseMatrix
import breeze.stats.distributions.MultivariateGaussian

object FilterOu {

  /**
    * Advance the distribution of the state using the OU process
    * @param svp the parameters of the stochastic volatility model
    * @param st the state at time t
    * @param dt the time difference to the next observation or point of interest
    * @param p the parameters of the DlM
    * @return the a-priori distribution of the latent state at time t + dt 
    */
  def advanceState(
    p: SvParameters)(
    st: KfState,
    dt: Double): KfState = {

    val variance = (p.sigmaEta / (2 * p.phi)) * (1 - exp(-2 * p.phi * dt))
    val identity = DenseMatrix.eye[Double](st.mt.size)

    st.copy(time = st.time + dt,
      at = st.mt map (m => p.mu + exp(p.phi * dt) * (m - p.mu)),
      rt = (identity * exp(-p.phi * dt)) * st.ct + (identity * variance))
  }

  def backwardStep(
    p:        SvParameters,
  )(kfState:  KfState,
    s:        SamplingState): SamplingState = {

    val dt = s.time - kfState.time
    val phi = exp(-p.phi * dt)
    val variance = (math.pow(p.sigmaEta, 2) / (2*p.phi)) * (1 - exp(-2*p.phi*dt))

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
