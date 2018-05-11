package dlm.model

import Dlm._
import breeze.linalg.{DenseVector, diag, DenseMatrix, sum, cholesky}
import breeze.stats.distributions._
import breeze.numerics._
import cats.data.Kleisli

object GibbsSampling extends App {
  case class State(
    p:     Parameters, 
    state: Vector[(Double, DenseVector[Double])]
  )

  def innerJoin[A, B](xs: Seq[A], ys: Seq[B])(pred: (A, B) => Boolean): Seq[(A, B)] = {
    for {
      x <- xs
      y <- ys if pred(x, y)
    } yield (x, y)
  }

  /**
    * Calculate the sum of squared differences between the one step forecast and the actual observation for each time
    * sum((y_t - f_t)^2)
    * @param f the observation matrix, a function from time => DenseMatrix[Double]
    * @param state an array containing the state sampled from the backward sampling algorithm
    * @param observations an array containing the actual observations of the data
    * @return the sum of squared differences between the one step forecast and the actual observation for each time
    */
  def observationSquaredDifference(
    f:            Double => DenseMatrix[Double],
    state:        Vector[(Double, DenseVector[Double])],
    observations: Vector[Data]) = {

    val ft = innerJoin(state, observations)((s, d) => s._1 == d.time).
      map { case ((time, x), y) => 
        val fm = KalmanFilter.missingF(f, time, y.observation)
        fm.t * x
      }

    val flatObservations = observations.map(_.observation).
      map(KalmanFilter.flattenObs)

    (flatObservations, ft).zipped.
      map { case (y, fr) => (y - fr) *:* (y - fr) }.
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
    f:            Double => DenseMatrix[Double],
    observations: Vector[Data])(s: State) = {

    val ssy = observationSquaredDifference(f, s.state, observations)

    val shape = prior.shape + observations.size * 0.5
    val rate = ssy.map(ss => prior.scale + ss * 0.5)

    val res = rate.map(r =>
      InverseGamma(shape, r).draw
    )

    Rand.always(s.copy(p = s.p.copy(v = diag(res))))
  }

  /**
    * Sample the diagonal system matrix for an irregularly observed 
    * DLM
    */
  def sampleSystemMatrix(
    prior: InverseGamma,
    g:     Double => DenseMatrix[Double])(s: State) = {
    
    val stateMean = s.state.tail

    // take the squared difference of x_t - g * x_{t-1} for t = 1 ... 0
    // add them all up
    val squaredSum = stateMean.zip(stateMean.tail).
    map { case (mt, mt1) =>
      val dt = mt1._1 - mt._1
      val diff = (mt1._2 - g(dt) * mt._2)
      (diff *:* diff) / dt }.
      reduce(_ + _)

    val shape = prior.shape + (s.state.size - 1) * 0.5
    val rate = squaredSum map (s => prior.scale + s * 0.5)

    val res = rate.map(r =>
      InverseGamma(shape, r).draw
    )

    Rand.always(s.copy(p = s.p.copy(w = diag(res))))
  }

  /**
    * Calculate the marginal likelihood of phi given
    * the values of the latent-state and other static parameters
    * @param state a sample of the latent state of an AR(1) DLM
    * @param p the static parameters of a DLM
    * @param phi autoregressive
    */
  def arlikelihood(
    state:  Vector[(Double, DenseVector[Double])],
    p:      Parameters,
    phi:    DenseVector[Double]): Double = {

    val n = state.length
    val ssa = (state zip state.tail).
      map { case (at, at1) => at1._2 - phi * at._2 }.
      map (x =>  x * x).
      reduce(_ + _)

    val det = sum(log(diag(cholesky(p.w))))

    - n * 0.5 * log(2 * math.Pi) + det - 0.5 * (p.w \ ssa).dot(ssa)
  }

  /**
    * Update an autoregressive model with a new value of the autoregressive
    * parameter
    */
  def updateModel(
    mod: Model,
    phi: Double*): Model = {

    mod.copy(g = (dt: Double) => new DenseMatrix(phi.size, 1, phi.toArray))
  }

  /**
    * Sample the autoregressive parameter with a Beta Prior
    * and proposal distribution
    * @param
    */
  def samplePhi(
    prior:  Beta,
    lambda: Double,
    tau:    Double,
    s:      State) = {

    val proposal = (phi: Double) => 
      new Beta(lambda * phi + tau, lambda * (1 - phi) + tau)

    val pos = (phi: Double) => 
      prior.logPdf(phi) + arlikelihood(s.state, s.p, DenseVector(phi))

    MarkovChain.Kernels.metropolisHastings(proposal)(pos)
  }

  /**
    * Perform forward filtering backward sampling
    */
  def ffbs(
    mod:          Model, 
    observations: Vector[Data])(s: State) = {

    for {
      newState <- SvdSampler.ffbs(mod, observations, s.p)
    } yield s.copy(state = newState.sortBy(_._1))
  }

  /**
    * A single step of a Gibbs Sampler
    * @param mod the model containing the definition of the observation matrix F_t and system evolution matrix G_t
    * @param priorV the prior distribution on the observation noise matrix, V
    * @param priorW the prior distribution on the system noise matrix, W
    * @param observations an array of Data containing the observed time series
    */
  def dinvGammaStep(
    mod:          Model, 
    priorV:       InverseGamma,
    priorW:       InverseGamma, 
    observations: Vector[Data]): Kleisli[Rand, State, State] = {

    Kleisli(sampleObservationMatrix(priorV, mod.f, observations)) compose
      Kleisli(ffbs(mod, observations)) compose
      Kleisli(sampleSystemMatrix(priorW, mod.g))
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
    observations: Vector[Data]) = {

    val initState = SvdSampler.ffbs(mod, observations, initParams).draw
    val init = State(initParams, initState)

    MarkovChain(init)(dinvGammaStep(mod, priorV, priorW, observations).run)
  }


  /**
    * Calculate the marginal likelihood for metropolis hastings
    */
  def likelihood(
    g: Double => DenseMatrix[Double]
  )(s: State): Double = {

    val n = s.state.length
    val ssa = (s.state zip s.state.tail).
      map { case (at, at1) =>
        val dt = at1._1 - at._1
        at1._2 - g(dt) * at._2 }.
      map (x =>  x * x).
      reduce(_ + _)

    val det: Double = sum(log(diag(cholesky(s.p.w))))
    - n * 0.5 * log(2 * math.Pi) + det - 0.5 * (s.p.w \ ssa).dot(ssa)
  }

  /**
    * A metropolis step for a DLM
    * @param mod a DLM model
    * @param proposal a symmetric proposal distribution for the parameters
    * of a DLM
    * @return a function from State => Rand[State] which performs a metropolis step
    * to be used in a Markov Chain
    */
  def metropStep(
    mod:      Model,
    proposal: Parameters => Rand[Parameters]) = {

    val prop = (s: State) => for {
      propP <- proposal(s.p)
    } yield s.copy(p = propP)

    MarkovChain.Kernels.metropolis(prop)(likelihood(mod.g))
  }

  /**
    * Use metropolis hastings to determine the initial state distribution x0 ~ N(m0, C0)
    * @param proposal a proposal distribution for the parameters of the initial state
    * @param mod a DLM model specification
    * @param priorV the prior distribution of the observation noise matrix
    * @param priorW the prior distribution of the system noise matrix
    * @param observations a vector of observations
    * @param gibbsState the current state of the Markov Chain
    */
  def gibbsMetropStep(
    proposal:     Parameters => Rand[Parameters],
    mod:          Model, 
    priorV:       InverseGamma,
    priorW:       InverseGamma, 
    observations: Vector[Data]) = {

    Kleisli(sampleObservationMatrix(priorV, mod.f, observations)) compose
      Kleisli(ffbs(mod, observations)) compose 
      Kleisli(sampleSystemMatrix(priorW, mod.g)) compose
      Kleisli(metropStep(mod, proposal))
  }

  /**
    * 
    */
  def metropSamples(
    proposal:     Parameters => Rand[Parameters],
    mod:          Model, 
    priorV:       InverseGamma, 
    priorW:       InverseGamma, 
    initParams:   Parameters, 
    observations: Vector[Data]) = {

    val initState = Smoothing.ffbs(mod, observations, initParams).draw
    val init = State(initParams, initState)

    MarkovChain(init)(gibbsMetropStep(proposal, mod, priorV, priorW, observations).run)
  }
}
