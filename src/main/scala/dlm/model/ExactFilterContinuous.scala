package dlm.model

import ContinuousTime._
import Dlm.Parameters
import breeze.linalg.{DenseVector, DenseMatrix}

object ExactFilter {
  def advanceState(
    g:    TimeIncrement => DenseMatrix[Double], 
    mt:   DenseVector[Double], 
    ct:   DenseMatrix[Double],
    dt:   TimeIncrement, 
    p:    Parameters) = {

    val at = g(dt) * mt
    val rt = g(dt) * ct * g(dt).t + (dt * p.w)

    (at, rt)
  }

  def step(
    mod:    Model, 
    p:      Parameters)
    (state: KalmanFilter.State,
    y:      Data): KalmanFilter.State = {

    val dt = y.time - state.time
    val (at, rt) = advanceState(mod.g, state.mt, state.ct, dt, p)
    val (ft, qt) = KalmanFilter.oneStepPrediction(mod.f, at, rt, y.time, p)
    val (mt, ct) = KalmanFilter.updateState(mod.f, at, rt, ft, qt, y, p)

    val ll = state.ll + KalmanFilter.conditionalLikelihood(ft, qt, y.observation)

    KalmanFilter.State(y.time, mt, ct, at, rt, Some(ft), Some(qt), ll)
  }

  def filter(
    mod:          Model, 
    observations: Array[Data], 
    p:            Parameters) = {

    val (at, rt) = advanceState(mod.g, p.m0, p.c0, 1.0, p)
    val init = KalmanFilter.State(observations.map(_.time).min - 1, 
      p.m0, p.c0,
      at, rt, None, None, 0.0)

    observations.scanLeft(init)(step(mod, p))
  }
}
