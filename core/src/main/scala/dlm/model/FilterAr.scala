package dlm.core.model

import breeze.linalg.{DenseMatrix, diag, svd}
import breeze.stats.distributions.{Gaussian, Rand}
/**
  * Filtering for the stochastic volatility model
  */
object FilterAr {
  case class FilterState(
    mt: Double,
    ct: Double,
    at: Double,
    rt: Double)

  def stepUni(p: SvParameters)(
    v:  Double,
    yo:  Option[Double],
    st: FilterState) = {

    val at = p.mu + p.phi * (st.mt - p.mu)
    val rt = p.phi * p.phi * st.ct + p.sigmaEta * p.sigmaEta

    yo match {
      case Some(y) =>
        val kt = rt / (rt + v)
        val et = y - at

        FilterState(at + kt * et, kt * v, at, rt)
      case None =>
        FilterState(at, rt, at, rt)
    }
  }

  /**
    * Univariate Kalman Filter for the stochastic volatility model with AR(1) state space
    */
  def filterUnivariate(
    ys: Vector[Option[Double]],
    vs: Vector[Double],
    p:  SvParameters) = {

    val (m0, c0) = (p.mu, p.sigmaEta * p.sigmaEta / (1 - p.phi * p.phi))
    val init = FilterState(m0, c0, m0, c0)

    (ys zip vs).scanLeft(init){ case (st, (y, v)) => stepUni(p)(v, y, st) }
  }

  case class SampleState(
    sample: Double,
    mt:  Double,
    ct:  Double,
    at1: Double,
    rt1: Double)

  def backStepUni(p: SvParameters)(
    fs: FilterState,
    ss: SampleState) = {

    val mean = fs.mt + (fs.ct * p.phi / ss.rt1) * (ss.sample - ss.at1)
    val cov = fs.ct - (fs.ct * fs.ct * p.phi * p.phi) / ss.rt1

    val sample = Gaussian(mean, math.sqrt(cov)).draw
    SampleState(sample, mean, cov, fs.at, fs.rt)
  }

  def univariateSample(
    p: SvParameters,
    fs: Vector[FilterState]) = {
    val last = fs.last
    val lastState = Gaussian(last.mt, math.sqrt(last.ct)).draw
    val init = SampleState(lastState, last.mt, last.ct, last.at, last.rt)
    Rand.always(fs.init.scanRight(init)(backStepUni(p)))
  }

  def ffbs(
    p: SvParameters,
    ys: Vector[Option[Double]],
    vs: Vector[Double]): Rand[Vector[SampleState]] = {

    val fs = filterUnivariate(ys, vs, p)
    univariateSample(p, fs)
  }

  def toKfState(
    ss: SampleState): FilterState = 
    FilterState(ss.mt, ss.ct, ss.at1, ss.rt1)

  def conditionalFilter(
    start: SampleState,
    p:     SvParameters,
    ys:    Vector[Option[Double]]) = {

    val v = math.Pi * math.Pi * 0.5
    val s0 = toKfState(start)
    ys.scanLeft(s0)((st, y) => stepUni(p)(v, y, st))
  }

  def conditionalSampler(
    end: SampleState,
    p:   SvParameters,
    fs:  Vector[FilterState]) = 
    Rand.always(fs.init.scanRight(end)(backStepUni(p)))

  def conditionalFfbs(
    start: SampleState,
    end:   SampleState,
    p:     SvParameters,
    ys:    Vector[Option[Double]]) = {

    val fs = conditionalFilter(start, p, ys)
    val sampled = conditionalSampler(end, p, fs.init)

    sampled.map(s => start +: s)
  }

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
    dt: Double) = st.copy(
      time = st.time + dt,
      at = st.mt map (m => p.mu + p.phi * (m - p.mu)),
      rt = st.ct map (c => p.phi * p.phi * c + p.sigmaEta * p.sigmaEta))

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
    filtered: Vector[SvdState]): Vector[SamplingState] = {

    val mod = Dlm.autoregressive(phi)
    val sqrtW = w map math.sqrt

    SvdSampler.sample(mod, filtered, sqrtW)
  }
}
