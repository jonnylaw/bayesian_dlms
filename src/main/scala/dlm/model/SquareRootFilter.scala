// package dlm.model

// import breeze.linalg._
// import breeze.stats.distributions._
// import Dlm._

// /**
//   * P = U D U^T Singular value decomposition of the covariance matrix
//   * U is a unit triangular matrix, a triangular matrix with 1s on the diagonal
//   * D is a diagonal matrix containing the eigen values of the covariance matrix
//   * We update the U and D matrices in the Prediction and Update Steps
//   */
// object SquareRootFilter {
//   case class State(
//     time:      Time,
//     stateMean: DenseVector[Double],
//     ut:         DenseMatrix[Double],
//     dt:         DenseVector[Double],
//     ll:        Double
//   )

//   def advanceState(
//     mod:  Model, 
//     mt:   DenseVector[Double],
//     ut:   DenseMatrix[Double],
//     dt:   DenseVector[Double],
//     time: Time, 
//     p:    Parameters): State = {

//     val at: DenseVector[Double] = mod.g(time) * mt
//     val ut: DenseMatrix[Double] = 
//     val dt: DenseVector[Double] = 

//     (at, ut, dt)
//   }

//   def oneStepPrediction(
//     mod:  Model,
//     at:   DenseVector[Double],
//     ut:   DenseMatrix[Double],
//     dt:   DenseVector[Double],
//     time: Time, 
//     p:    Parameters) = {

//     val ft: DenseVector[Double] = mod.f(time).t * at
//     val qt: DenseMatrix[Double] = mod.f(time) *  * mod.f(time).t + p.v

//     (ft, qt)
//   }

//   /**
//     * @param 
//     */
//   def updateState(
//     mod:       Model, 
//     at:        DenseVector[Double], 
//     ut:        DenseMatrix[Double],
//     dt:        DenseVector[Double], 
//     predicted: Observation, 
//     qt:        DenseMatrix[Double],
//     y:         Data, 
//     p:         Parameters) = y.observation match {
//     case Some(obs) =>
//       val time = y.time
//       val residual = obs - predicted
//       val kalman_gain = 
//       val mt = 

//       val ut1 = 
//       val dt1 = 

//       (mt, ut1, dt1)
//     case None =>
//       (at, ut, dt)
//   }

//   def conditionalLikelihood(
//     ft: Observation, 
//     qt: DenseMatrix[Double], 
//     data: Option[Observation]) = data match {
//     case Some(y) =>
//       MultivariateGaussian(ft, qt).logPdf(y)
//     case None => 0.0
//   }

//   def stepKalmanFilter(
//     mod: Model, p: Parameters)(state: State, y: Data): State = {

//     val (at, ut, dt) = advanceState(mod, state.stateMean, state.ut, state.dt, y.time, p)
//     val (ft, qt) = oneStepPrediction(mod, at, ut, dt, y.time, p)
//     val (mt, ut1, dt1) = updateState(mod, at, ut, dt, ft, qt, y, p)

//     val ll = state.ll + conditionalLikelihood(ft, qt, y.observation)

//     State(y.time, state_posterior, mt, ut1, dt1, Some(ft), Some(qt), ll)
//   }

//   /**
//     * Run the Kalman Filter over an array of data
//     */
//   def kalmanFilter(mod: Model, observations: Array[Data], p: Parameters) = {
//     val initState = MultivariateGaussian(p.m0, p.c0)
//     val init: KfState = KfState(
//       observations.head.time,
//       initState,
//       advanceState(mod, initState, 0, p),
//       None, None, 0.0)

//     observations.scanLeft(init)(stepKalmanFilter(mod, p)).drop(1)
//   }
// }
