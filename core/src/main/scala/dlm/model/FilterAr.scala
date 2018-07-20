package core.dlm.model

import breeze.linalg.{DenseMatrix, diag, svd}

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

  /**
    * Advance the state mean and variance to the a-priori
    * value of the state at time t
    * @param st the kalman filter state
    * @param dt the time difference between observations (1.0)
    * @param p the parameters of the SV Model
    * @return the a-priori mean and covariance of the state at time t
    */
  def advanceStateSvd(
    p:  SvParameters)(
    st: SvdState,
    dt: Double): SvdState = {
    
    if (dt == 0) {
      st
    } else {
      val identity = DenseMatrix.eye[Double](st.mt.size)
      val g = identity * p.phi
      val sqrtW = identity * p.sigmaEta
      val rt = DenseMatrix.vertcat(diag(st.dc) * st.uc.t * g.t, sqrtW *:* math.sqrt(dt))
      val root = svd(rt)

      st.copy(at = st.mt map (m => p.mu + p.phi * (m - p.mu)),
        dr = root.singularValues,
        ur = root.rightVectors.t)
    }
  }

  def backStep(params: SvParameters) = {
    val mod = Dlm.autoregressive(params.phi)
    val p = StochasticVolatility.ar1DlmParams(params)
    Smoothing.step(mod, p.w) _
  }

  def sample(
    params: SvParameters,
    filtered: Vector[KfState]) = {

    val mod = Dlm.autoregressive(params.phi)
    val p = StochasticVolatility.ar1DlmParams(params)
    Smoothing.sample(mod, filtered, Smoothing.step(mod, p.w))
  }

  def sampleSvd(
    w:        DenseMatrix[Double],
    phi:      Double,
    filtered: Vector[SvdState]) = {

    val mod = Dlm.autoregressive(phi)
    val sqrtW = w map math.sqrt

    SvdSampler.sample(mod, filtered, sqrtW)
  }
}
