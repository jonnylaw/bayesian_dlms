package dlm.model

import Dlm._
import Smoothing._
import KalmanFilter._
import cats.implicits._
import breeze.linalg.{DenseVector, diag, DenseMatrix}
import breeze.stats.distributions.{Rand, MarkovChain}

object GibbsSampling extends App {
  case class State(
    p:     Parameters, 
    state: Array[(Time, DenseVector[Double])]
  )

  /**
    * Calculate the sum of squared differences between the one step forecast and the actual observation for each time
    * sum((y_t - f_t)^2)
    * @param f the observation matrix, a function from time => DenseMatrix[Double]
    * @param state an array containing the state sampled from the backward sampling algorithm
    * @param observations an array containing the actual observations of the data
    * @return the sum of squared differences between the one step forecast and the actual observation for each time
    */
  def observationSquaredDifference(
    f:            Time => DenseMatrix[Double],
    state:        Array[(Time, DenseVector[Double])],
    observations: Array[Data]) = {

    val forecast = state.sortBy(_._1).tail.
      map { case (time, x) => f(time).t * x }

    (observations.sortBy(_.time).map(_.observation), forecast).zipped.
      map { 
        case (Some(y), fr) => (y - fr) *:* (y - fr)
        case (None, f) => DenseVector.zeros[Double](f.size)
      }.
      reduce(_ + _)
  }

  /**
    * Sample the (diagonal) observation noise covariance matrix from an Inverse Gamma distribution
    * @param prior an Inverse Gamma prior distribution for each variance element of the observation matrix
    * @param mod the DLM specification
    * @param state a sample of the DLM state
    * @param observations the observed values of the time series
    * @return the posterior distribution over the diagonal observation matrix  
    */
  def sampleObservationMatrix(
    prior:        InverseGamma,
    f:            Time => DenseMatrix[Double],
    state:        Array[(Time, DenseVector[Double])],
    observations: Array[Data]): Rand[DenseMatrix[Double]] = {

    val ssy = observationSquaredDifference(f, state, observations)

    val shape = prior.shape + observations.size * 0.5
    val rate = ssy.map(ss => prior.scale + ss * 0.5)

    val res = rate.map(r =>
      InverseGamma(shape, r).draw
    )

    Rand.always(diag(res))
  }

  def stateSquaredDifference(
    g:            Time => DenseMatrix[Double],
    state:        Array[(Time, DenseVector[Double])]) = {

    // sort the state by time
    val sortState = state.sortBy(_._1)

    // take every element of the state but the last, x_0,...,x_{t-1}
    // and advance them according to the model
    val advanceState = sortState.init.
      map { case (time, x) => g(time) * x }
    
    // take every element by the first, x_1,...,x_t
    val stateMean = sortState.map(_._2).tail

    // take the squared difference of x_t - g * x_{t-1} for t = 1 ... 0
    // add them all up
    (stateMean, advanceState).zipped.
      map { case (mt, at) => (mt - at) *:* (mt - at) }.
      reduce(_ + _)
  }

  /**
    * Sample the (diagonal) system noise covariance matrix from an inverse gamma 
    * @param prior 
    * @param g
    * @param state
    */
  def sampleSystemMatrix(
    prior: InverseGamma,
    g:     Time => DenseMatrix[Double],
    state: Array[(Time, DenseVector[Double])]): Rand[DenseMatrix[Double]] = {

    val sstheta = stateSquaredDifference(g, state)
    val shape = prior.shape + (state.size - 1) * 0.5
    val rate = sstheta map (s => prior.scale + s * 0.5)

    val res = rate.map(r =>
      InverseGamma(shape, r).draw
    )

    Rand.always(diag(res))
  }

  /**
    * Calculate the lagged difference between items in a Seq
    * @param xs a sequence of numeric values
    * @return a sequence of numeric values containing the once lagged difference
    */
  def diff[A](xs: Seq[A])(implicit A: Numeric[A]): Seq[A] = {
    (xs, xs.tail).zipped.map { case (x, x1) => A.minus(x1, x) }
  }

  /**
    * Sample the diagonal system matrix for an irregularly observed 
    * DLM
    */
  def sampleSystemMatrixCont(
    prior: InverseGamma,
    g:     TimeIncrement => DenseMatrix[Double],
    state: Array[(Time, DenseVector[Double])]): Rand[DenseMatrix[Double]] = {
    
    val sortedState = state.sortBy(_._1)
    val times = sortedState.map(_._1)
    val deltas = diff(times)
    val advanceState = (deltas, sortedState.init.map(_._2)).
      zipped.
      map { case (dt, x) => g(dt) * x }

    val stateMean = sortedState.map(_._2).tail

    // take the squared difference of x_t - g * x_{t-1} for t = 1 ... 0
    // add them all up
    val squaredSum = (deltas zip stateMean zip advanceState).
      map { case ((dt, mt), at) => ((mt - at) *:* (mt - at)) / dt }.
      reduce(_ + _)

    val shape = prior.shape + (state.size - 1) * 0.5
    val rate = squaredSum map (s => prior.scale + s * 0.5)

    val res = rate.map(r =>
      InverseGamma(shape, r).draw
    )

    Rand.always(diag(res))
  }

  /**
    * Sample state
    */
  def sampleState(
    mod:          Model, 
    observations: Array[Data], 
    p:            Parameters) = {

    val filtered = kalmanFilter(mod, observations, p)
    backwardSampling(mod, filtered, p)
  }

  /**
    * A single step of a Gibbs Sampler
    */
  def dinvGammaStep(
    mod:          Model, 
    priorV:       InverseGamma,
    priorW:       InverseGamma, 
    observations: Array[Data])(gibbsState: State) = {

    for {
      obs <- sampleObservationMatrix(priorV, mod.f, gibbsState.state, observations)
      state = sampleState(mod, observations, Parameters(obs, gibbsState.p.w, gibbsState.p.m0, gibbsState.p.c0))
      system <- sampleSystemMatrix(priorW, mod.g, state)
    } yield State(Parameters(obs, system, gibbsState.p.m0, gibbsState.p.c0), state)
  }

  /**
    * Return a Markov chain using Gibbs Sampling to determine the values of the system and 
    * observation noise covariance matrices, W and V
    * @param mod the model containing the definition of the observation matrix F_t and system evolution matrix G_t
    * @param priorV the prior distribution on the observation noise matrix, V
    * @param priorW the prior distribution on the system noise matrix, W
    * @param initParams the initial parameters of the Markov Chain
    * @param observations an array of Data containing the observed time series
    * @return a Process 
    */
  def sample(
    mod:          Model, 
    priorV:       InverseGamma, 
    priorW:       InverseGamma, 
    initParams:   Parameters, 
    observations: Array[Data]) = {

    val initState = sampleState(mod, observations, initParams)
    val init = State(initParams, initState)

    MarkovChain(init)(dinvGammaStep(mod, priorV, priorW, observations))
  }

  def metropStep(
    mod:          Model, 
    observations: Array[Data],
    proposal:     Parameters => Rand[Parameters]) = {

    MarkovChain.Kernels.metropolis(proposal)(KalmanFilter.logLikelihood(mod, observations))
  }

  def gibbsMetropStep(
    proposal:     Parameters => Rand[Parameters],
    mod:          Model, 
    priorV:       InverseGamma,
    priorW:       InverseGamma, 
    observations: Array[Data])(gibbsState: State) = {

    for {
      obs <- sampleObservationMatrix(priorV, mod.f, gibbsState.state, observations)
      state = sampleState(mod, observations, gibbsState.p.copy(v = obs))
      system <- sampleSystemMatrix(priorW, mod.g, state)
      p = Parameters(obs, system, gibbsState.p.m0, gibbsState.p.c0)
      newP <- metropStep(mod, observations, proposal)(p)
    } yield State(newP, state)
  }

  def metropSamples(
    proposal:     Parameters => Rand[Parameters],
    mod:          Model, 
    priorV:       InverseGamma, 
    priorW:       InverseGamma, 
    initParams:   Parameters, 
    observations: Array[Data]) = {

    val initState = sampleState(mod, observations, initParams)
    val init = State(initParams, initState)

    MarkovChain(init)(gibbsMetropStep(proposal, mod, priorV, priorW, observations))
  }

  /**
    * A single step
    */
  def stepContinuous(
    mod:          ContinuousTime.Model,
    priorV:       InverseGamma,
    priorW:       InverseGamma, 
    observations: Array[Data]
  )(s: State): Rand[State] = {
    for {
      state <- ExactBackSample.ffbs(mod, observations, s.p)
      w <- sampleSystemMatrixCont(priorW, mod.g, state)
      v <- sampleObservationMatrix(priorV, mod.f, state, observations)
    } yield State(Dlm.Parameters(v, w, s.p.m0, s.p.c0), state)
  }

  def sampleContinuous(
    mod:          ContinuousTime.Model,
    priorV:       InverseGamma,
    priorW:       InverseGamma, 
    observations: Array[Data],
    initParams:   Parameters
  ) = {
    val initState = ExactBackSample.ffbs(mod, observations, initParams).draw
    val init = State(initParams, initState)

    MarkovChain(init)(stepContinuous(mod, priorV, priorW, observations))
  }

}
