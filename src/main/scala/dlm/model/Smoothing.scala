package dlm.model

import Dlm._
import breeze.linalg.{inv, DenseVector, DenseMatrix}
import breeze.stats.distributions.{MultivariateGaussian, Rand}

object Smoothing {
  case class SmoothingState(
    time:       Time, 
    mean:       DenseVector[Double], 
    covariance: DenseMatrix[Double], 
    at1:        DenseVector[Double], 
    rt1:        DenseMatrix[Double])

  /**
    * A single step in the backwards smoother
    * Requires that a Kalman Filter has been run on the model
    * @param mod a DLM model specification
    * @param state the state at time t + 1
    * @return 
    */
  def smoothStep(
    mod:     Model)
    (state:  SmoothingState,
    kfState: KalmanFilter.State) = {

    // extract elements from kalman state
    val time = kfState.time
    val dt = state.time - kfState.time
    val mt = kfState.mt
    val ct = kfState.ct
    val at1 = state.at1
    val rt1 = state.rt1

    val cgrinv = (rt1.t \ (mod.g(dt) * ct.t)).t

    // calculate the updated mean 
    val mean = mt + cgrinv * (state.mean - at1)

    // val n = w.cols
    // val identity = DenseMatrix.eye[Double](n)
    // val diff = identity - cgrinv * mod.g(dt)
    // val covariance = diff * ct * diff.t + cgrinv * w * dt * cgrinv.t

    // calculate the updated covariance
    val covariance = ct - cgrinv * (rt1 - state.covariance) * cgrinv

    SmoothingState(time, mean, covariance, kfState.at, kfState.rt)
  }

  /**
    * Learn the distribution of the latent state (the smoothing distribution)
    * p(x_{1:T} | y_{1:T}) of a fully specified DLM with observations available
    * for all time
    * @param mod a DLM model specification
    * @param kfState the output of a Kalman Filter
    * @return
    */
  def backwardsSmoother(
    mod:      Model)
    (kfState: Array[KalmanFilter.State]) = {

    val sortedState = kfState.sortWith(_.time > _.time)
    val last = sortedState.head
    val lastTime = last.time
    val init = SmoothingState(lastTime, last.mt, last.ct, last.at, last.rt)

    sortedState.tail.scanLeft(init)(smoothStep(mod)).
      sortBy(_.time)
  }

  case class SamplingState(
    time:       Time, 
    sample:     DenseVector[Double],
    at1:        DenseVector[Double], 
    rt1:        DenseMatrix[Double])

  /**
    * Copies the lower triangular portion of a matrix to the upper triangle
    */
  def makeSymmetric(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    val n = m.cols
    DenseMatrix.tabulate(n, n){ case (i, j) =>
      if (i > j) {
        m(i, j)
      } else if (i < j) {
        m(j, i)
      } else {
        m(i, i)
      }
    }
  }

  def step(
    mod:      Model, 
    w:        DenseMatrix[Double])
    (state:   Smoothing.SamplingState, 
     kfState: KalmanFilter.State) = {

    // extract elements from kalman state
    val time = kfState.time
    val dt = state.time - kfState.time
    val mt = kfState.mt
    val ct = kfState.ct
    val at1 = state.at1
    val rt1 = state.rt1

    // more efficient than inverting rt, equivalent to C * G.t * inv(R)
    val cgrinv = (rt1.t \ (mod.g(dt) * ct.t)).t

    // calculate the updated mean
    val mean = mt + cgrinv * (state.sample - at1)

    // calculate the updated covariance
    val n = w.cols
    val identity = DenseMatrix.eye[Double](n)
    val diff = identity - cgrinv * mod.g(dt)
    val covariance = diff * ct * diff.t + cgrinv * w * dt * cgrinv.t

    Smoothing.SamplingState(kfState.time, 
      MultivariateGaussianSvd(mean, covariance).draw, 
      kfState.at, 
      kfState.rt)
  }

  def sample(
    mod:     Model,
    kfState: Array[KalmanFilter.State], 
    w:       DenseMatrix[Double]) = {

    // sort the state in reverse order
    val sortedState = kfState.sortWith(_.time > _.time)

    // extract the final state
    val last = sortedState.head
    val lastTime = last.time
    val lastState = MultivariateGaussianSvd(last.mt, last.ct).draw
    val initState = Smoothing.SamplingState(lastTime, lastState, last.at, last.rt)

    sortedState.tail.
      scanLeft(initState)(step(mod, w)).
      sortBy(_.time).map(a => (a.time, a.sample))
  }

  /**
    * Forward filtering backward sampling for a DLM
    */
  def ffbs(
    mod:          Model,
    observations: Array[Data],
    p:            Dlm.Parameters) = {

    val filtered = KalmanFilter.filter(mod, observations, p)
    Rand.always(sample(mod, filtered, p.w))
  }
}
