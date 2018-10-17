package dlm.core.model

import math.exp
import breeze.stats.distributions.{Rand, Gaussian}

object FilterOu {
  def stepUni(p: SvParameters)(
    v:  Double,
    t:  Double,
    yo: Option[Double],
    st: FilterAr.FilterState) = {

    val dt = t - st.time
    val variance = (math.pow(p.sigmaEta, 2) * (1 - exp(-2*p.phi*dt))) / (2*p.phi)

    val at = p.mu + exp(-p.phi * dt) * (st.mt - p.mu)
    val rt = exp(-2 * p.phi * dt) * st.ct + variance

    yo match {
      case Some(y) =>
        val kt = rt / (rt + v)
        val et = y - at

        FilterAr.FilterState(t, at + kt * et, kt * v, at, rt)
      case None =>
        FilterAr.FilterState(t, at, rt, at, rt)
    }
  }

  /**
    * Univariate Kalman Filter for the stochastic volatility model 
    * with OU state space
    * @param ys 
    */
  def filterUnivariate(
    ys: Vector[(Double, Option[Double])],
    vs: Vector[Double],
    p:  SvParameters) = {

    val (m0, c0) = (p.mu, p.sigmaEta * p.sigmaEta / p.phi * p.phi)
    val t0 = ys.map(_._1).head
    val init = FilterAr.FilterState(t0, m0, c0, m0, c0)

    (ys zip vs).scanLeft(init){ case (st, ((t, y), v)) =>
      stepUni(p)(v, t, y, st) }
  }

 def backStepUni(p: SvParameters)(
    fs: FilterAr.FilterState,
    ss: FilterAr.SampleState) = {

    val dt = ss.time - fs.time
    val phi = (dt: Double) => exp(-p.phi * dt)

    val mean = fs.mt + (fs.ct * phi(dt) / ss.rt1) * (ss.sample - ss.at1)
    val cov = fs.ct - (math.pow(fs.ct, 2) * math.pow(phi(dt), 2)) / ss.rt1

    val sample = Gaussian(mean, math.sqrt(cov)).draw
    FilterAr.SampleState(fs.time, sample, fs.mt, fs.ct, fs.at, fs.rt)
  }

  def univariateSample(
    p: SvParameters,
    fs: Vector[FilterAr.FilterState]) = {
    val last = fs.last
    val lastState = Gaussian(last.mt, math.sqrt(last.ct)).draw
    val init = FilterAr.SampleState(last.time, lastState,
      last.mt, last.ct, last.at, last.rt)
    Rand.always(fs.init.scanRight(init)(backStepUni(p)))
  }

  def ffbs(
    p: SvParameters,
    ys: Vector[(Double, Option[Double])],
    vs: Vector[Double]): Rand[Vector[FilterAr.SampleState]] = {

    val fs = filterUnivariate(ys, vs, p)
    univariateSample(p, fs)
  }

  def toKfState(
    ss: FilterAr.SampleState): FilterAr.FilterState = 
    FilterAr.FilterState(ss.time, ss.mean, ss.cov, ss.at1, ss.rt1)

  def conditionalFilter(
    start: FilterAr.SampleState,
    p:     SvParameters,
    ys:    Vector[(Double, Option[Double])]) = {

    val v = math.Pi * math.Pi * 0.5
    val s0 = toKfState(start)
    ys.scanLeft(s0){ case (st, (t, y)) => stepUni(p)(v, t, y, st) }
  }

  def conditionalSampler(
    end: FilterAr.SampleState,
    p:   SvParameters,
    fs:  Vector[FilterAr.FilterState]) = 
    Rand.always(fs.init.scanRight(end)(backStepUni(p)))

  def conditionalFfbs(
    start: FilterAr.SampleState,
    end:   FilterAr.SampleState,
    p:     SvParameters,
    ys:    Vector[(Double, Option[Double])]) = {

    val fs = conditionalFilter(start, p, ys)
    val sampled = conditionalSampler(end, p, fs.init)

    sampled.map(s => start +: s)
  }
}
