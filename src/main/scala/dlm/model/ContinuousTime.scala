package dlm.model

import breeze.linalg._
import breeze.stats.distributions._

object ContinuousTime {
  /**
    * A continuous time model has a G matrix which depends on the 
    * time increment between latent-state realisations
    */
  case class Model(
    observation: (DenseVector[Double], DenseMatrix[Double]) => Rand[DenseVector[Double]],
    f: Time => DenseMatrix[Double],
    g: TimeIncrement => DenseMatrix[Double],
    conditionalLikelihood: (Dlm.Parameters) => (Observation, DenseVector[Double]) => Double
  )

  def angle(period: Int)(dt: TimeIncrement): Double = {
    2 * math.Pi * (dt % period) / period
  }

  def seasonalG(
    period:    Int,
    harmonics: Int)(
    dt:        TimeIncrement): DenseMatrix[Double] = {

    val matrices = (delta: TimeIncrement) => (1 to harmonics).
      map(h => Dlm.rotationMatrix(h * angle(period)(delta)))

    matrices(dt).reduce(Dlm.blockDiagonal)
  }

  def dglm2Model(mod: Dglm.Model, contG: TimeIncrement => DenseMatrix[Double]) = {
    Model(
      mod.observation,
      (t: Time) => mod.f(t),
      contG,
      mod.conditionalLikelihood
    )
  }

  def dlm2Model(mod: Dlm.Model, contG: TimeIncrement => DenseMatrix[Double]) = {
    Model(
      (x: DenseVector[Double], v: DenseMatrix[Double]) => MultivariateGaussianSvd(x, v),
      (t: Time) => mod.f(t),
      contG,
      (p: Dlm.Parameters) => (x: DenseVector[Double], y: DenseVector[Double]) => MultivariateGaussianSvd(x, p.v).logPdf(y)
    )
  }

  def simulate(times: Iterable[Double], mod: Model, p: Dlm.Parameters) = {
    val init = (times.head, MultivariateGaussianSvd(p.m0, p.c0).draw)

    val state = times.tail.scanLeft(init) { (x, t) =>
      val dt = t - x._1
      (t, MultivariateGaussianSvd(mod.g(dt) * x._2, p.w * dt).draw)
    }

    state.map { case (t, x) => 
      (Data(t, Some(MultivariateGaussianSvd(mod.f(t).t * x, p.v).draw)), x) 
    }
  }


  /**
    * Perform a single forecast step, equivalent to performing the Kalman Filter
    * Without an observation of the process
    * @param mod a DLM specification
    * @param time the current time 
    * @param mt the mean of the latent state at time t
    * @param ct the variance of the latent state at time t
    * @param p the parameters of the DLM
    */
  def stepForecast(
    mod:  Model,
    time: Time,
    mt:   DenseVector[Double], 
    ct:   DenseMatrix[Double],
    p:    Dlm.Parameters) = {

    val (at, rt) = ExactFilter.advanceState(mod.g, mt, ct, time, p)
    val (ft, qt) = KalmanFilter.oneStepPrediction(mod.f, at, rt, time, p)

    (time, at, rt, ft, qt)
  }

  /**
    * Forecast a DLM from a state
    */
  def forecast(
    mod:  Model, 
    mt:   DenseVector[Double], 
    ct:   DenseMatrix[Double],
    time: Time,
    p:    Dlm.Parameters) = {

    val (ft, qt) = KalmanFilter.oneStepPrediction(mod.f, mt, ct, time, p)

    Stream.iterate((time, mt, ct, ft, qt)){ 
      case (t, m, c, _, _) => stepForecast(mod, t + 1, m, c, p) }.
      map(a => (a._1, a._4.data(0), a._5.data(0)))
  }
}
