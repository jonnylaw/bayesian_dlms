// package dlm.model

// import ContinuousTime._
// import Dlm.Parameters
// import breeze.linalg.{DenseVector, DenseMatrix}
// import breeze.stats.distributions.Rand
// import cats._
// import cats.implicits._

// object ExactFilter {
//   def advanceState(
//     g:    TimeIncrement => DenseMatrix[Double], 
//     mt:   DenseVector[Double], 
//     ct:   DenseMatrix[Double],
//     dt:   TimeIncrement, 
//     w:    DenseMatrix[Double]): (DenseVector[Double], DenseMatrix[Double]) = {

//     val at = g(dt) * mt
//     val rt = g(dt) * ct * g(dt).t + (dt * w)

//     (at, rt)
//   }

//   def step(
//     mod:    Model, 
//     p:      Parameters)
//     (state: KalmanFilter.State,
//     y:      Data): KalmanFilter.State = {

//     val dt = y.time - state.time
//     val (at, rt) = advanceState(mod.g, state.mt, state.ct, dt, p.w)
//     val (ft, qt) = KalmanFilter.oneStepPrediction(mod.f, at, rt, y.time, p.v)
//     val (mt, ct) = KalmanFilter.updateState(mod.f, at, rt, ft, qt, y, p.v)

//     val ll = state.ll + KalmanFilter.conditionalLikelihood(ft, qt, y.observation)

//     KalmanFilter.State(y.time, mt, ct, at, rt, Some(ft), Some(qt), ll)
//   }

//   def filter(
//     mod:          Model, 
//     observations: Array[Data], 
//     p:            Parameters) = {

//     val (at, rt) = advanceState(mod.g, p.m0, p.c0, 1.0, p.w)
//     val init = KalmanFilter.State(observations.map(_.time).min - 1, 
//       p.m0, p.c0,
//       at, rt, None, None, 0.0)

//     observations.scanLeft(init)(step(mod, p))
//   }

//   def logLikelihood(
//     mod:          Model,
//     observations: Array[Data])
//     (p:            Parameters): Double = {

//     val (at, rt) = advanceState(mod.g, p.m0, p.c0, 1.0, p.w)
//     val init = KalmanFilter.State(observations.map(_.time).min - 1, 
//       p.m0, p.c0,
//       at, rt, None, None, 0.0)

//     observations.foldLeft(init)(step(mod, p)).ll
//   }
// }

// object ExactBackSample {

//   def step(
//     mod:      Model, 
//     p:        Parameters)
//     (state:   Smoothing.SamplingState, 
//      kfState: KalmanFilter.State) = {

//     // extract elements from kalman state
//     val time = kfState.time
//     val dt = state.time - kfState.time
//     val mt = kfState.mt
//     val ct = kfState.ct
//     val at1 = state.at1
//     val rt1 = state.rt1

//     // more efficient than inverting rt, equivalent to C * G.t * inv(R)
//     val cgrinv = (rt1.t \ (mod.g(dt) * ct.t)).t

//     // calculate the updated mean
//     val mean = mt + cgrinv * (state.sample - at1)

//     // calculate the updated covariance
//     val n = p.w.cols
//     val identity = DenseMatrix.eye[Double](n)
//     val diff = identity - cgrinv * mod.g(dt)
//     val covariance = diff * ct * diff.t + cgrinv * p.w * dt * cgrinv.t

//     Smoothing.SamplingState(kfState.time, 
//       MultivariateGaussianSvd(mean, covariance).draw, 
//       kfState.at, 
//       kfState.rt)
//   }

//   def sample(
//     mod:     Model,
//     kfState: Array[KalmanFilter.State], 
//     p:       Parameters) = {

//     // sort the state in reverse order
//     val sortedState = kfState.sortWith(_.time > _.time)

//     // extract the final state
//     val last = sortedState.head
//     val lastTime = last.time
//     val lastState = MultivariateGaussianSvd(last.mt, last.ct).draw
//     val initState = Smoothing.SamplingState(lastTime, lastState, last.at, last.rt)

//     sortedState.tail.
//       scanLeft(initState)(step(mod, p)).
//       sortBy(_.time).map(a => (a.time, a.sample))
//   }

//   /**
//     * Forward filtering backward sampling for a Continuous time DLM
//     */
//   def ffbs(
//     mod:          Model,
//     observations: Array[Data],
//     p:            Dlm.Parameters) = {

//     val filtered = ExactFilter.filter(mod, observations, p)
//     Rand.always(ExactBackSample.sample(mod, filtered, p))
//   }
// }
