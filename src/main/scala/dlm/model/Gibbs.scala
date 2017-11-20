package dlm.model

import Dlm._
import cats.implicits._
import breeze.linalg.{DenseVector, diag, DenseMatrix}
import breeze.stats.distributions.{Rand, MarkovChain, Gamma}

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
  def sampleSystemMatrix(
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
    * A single step of a Gibbs Sampler
    */
  def dinvGammaStep(
    mod:          Model, 
    priorV:       InverseGamma,
    priorW:       InverseGamma, 
    observations: Array[Data])(gibbsState: State) = {

    for {
      obs <- sampleObservationMatrix(priorV, mod.f, gibbsState.state, observations)
      state <- Smoothing.ffbs(mod, observations, gibbsState.p.copy(v = obs))
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

    val initState = Smoothing.ffbs(mod, observations, initParams).draw
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
      state <- Smoothing.ffbs(mod, observations, gibbsState.p.copy(v = obs))
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

    val initState = Smoothing.ffbs(mod, observations, initParams).draw
    val init = State(initParams, initState)

    MarkovChain(init)(gibbsMetropStep(proposal, mod, priorV, priorW, observations))
  }

  /**
    * Sample the variances of the Normal distribution 
    * These are auxilliary variables required when calculating 
    * the one-step prediction in the Kalman Filter
    * @param ys an array of observations of length N
    * @param f the observation matrix
    * @param state a sample from the state, using sampleStateT
    * @param dof the degrees of freedom of the Student's t-distribution
    * @param scale the (square of the) scale of the Student's t-distribution
    * @return a Rand distribution over the list of N variances
    */
  def sampleVariancesT(
    ys:    Array[Data],
    f:     ObservationMatrix,
    state: Array[(Time, DenseVector[Double])],
    dof:   Int,
    scale: Double
  ): Rand[List[Double]] = {
    val alpha = (dof + 1) * 0.5

    val diff = (ys.sortBy(_.time).map(_.observation), 
      state.sortBy(_._1).tail).zipped.
      map {
        case (Some(y), (time, x)) => (y - f(time).t * x) *:* (y - f(time).t * x)
        case (None, x) => DenseVector.zeros[Double](x._2.size)
      }

    val beta = diff.map(d => (dof * scale * 0.5) + d(0) * 0.5).toList

    beta.traverse(b => InverseGamma(alpha, b): Rand[Double])
  }

  /**
    * Sample the (square of the) scale of the Student's t distribution
    * @param
    */
  def sampleScaleT(
    variances: List[Double],
    dof:       Int): Rand[Double] = {
    val t = variances.size
  
    val shape = t * dof * 0.5 + 1
    val rate = dof * 0.5 * variances.map(1.0/_).sum
    val scale = 1 / rate

    Gamma(shape, scale)
  }

  /**
    * Sample the state, incorporating the drawn variances for each observation
    * @param 
    */
  def sampleStateT(
    variances:    List[Double],
    params:       Dlm.Parameters,
    mod:          Dlm.Model,
    observations: Array[Data]
  ) = {
    // create a list of parameters with the variance in them
    val ps = variances.map(vi => params.copy(v = DenseMatrix(vi))).toArray

    def kalmanStep(p: Dlm.Parameters) = KalmanFilter.step(mod, p) _

    val (at, rt) = KalmanFilter.advanceState(mod.g, params.m0, 
      params.c0, 0, params.w)

    val init = KalmanFilter.State(
      observations.map(_.time).min - 1, 
      params.m0, params.c0, at, rt, None, None, 0.0)

    // fold over the list of variances and the observations
    val filtered = ps.zip(observations).
      scanLeft(init){ case (s, (p, y)) => kalmanStep(p)(s, y) }

    Rand.always(Smoothing.sample(mod, filtered, params.w))
  }

  /**
    * The state of the Markov chain for the Student's t-distribution
    * gibbs sampler
    */
  case class StudentTState(
    p:         Dlm.Parameters,
    variances: List[Double],
    state:     Array[(Time, DenseVector[Double])]
  )

  /**
    * A single step of the Student t-distribution Gibbs Sampler
    */
  def studentTStep(
    dof:    Int, 
    data:   Array[Data],
    priorW: InverseGamma,
    mod:    Dglm.Model
  )(state: StudentTState) = {
    val dlm = Model(mod.f, mod.g)

    for {
      latentState <- sampleStateT(state.variances, state.p, dlm, data)
      newW <- sampleSystemMatrix(priorW, mod.g, latentState)
      variances <- sampleVariancesT(data, mod.f, latentState, dof, state.p.v(0,0))
      scale <- GibbsSampling.sampleScaleT(variances, dof)
    } yield StudentTState(
      state.p.copy(v = DenseMatrix(scale), w = newW),
      variances, 
      latentState
    )
  }

  /**
    * Perform Gibbs Sampling for the Student t-distributed model
    */
  def studentT(
    dof:    Int,
    data:   Array[Data],
    priorW: InverseGamma,
    mod:    Dglm.Model,
    params: Dlm.Parameters
  ) = {
    val dlm = Model(mod.f, mod.g)
    val initVariances = List.fill(data.size)(1.0)
    val initState = GibbsSampling.sampleStateT(initVariances, params, dlm, data)
    val init = StudentTState(params, initVariances, initState.draw)

    MarkovChain(init)(studentTStep(dof, data, priorW, mod))
  }
}
