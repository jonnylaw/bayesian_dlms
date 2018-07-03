package core.dlm.model

import Dlm._
import breeze.linalg.{DenseVector, diag, DenseMatrix, sum, cholesky}
import breeze.stats.distributions._
import breeze.numerics._

object GibbsSampling extends App {
  case class State(
      p: DlmParameters,
      state: Vector[(Double, DenseVector[Double])]
  )

  /**
    * Sample the (diagonal) observation noise covariance matrix
    * from an Inverse Gamma distribution
    * @param prior an Inverse Gamma prior distribution for each
    * variance element of the observation matrix
    * @param mod the DLM specification
    * @param theta a sample of the DLM state
    * @param ys the observed values of the time series
    * @return the posterior distribution over the diagonal observation matrix
    */
  def sampleObservationMatrix(
    prior: InverseGamma,
    f: Double => DenseMatrix[Double],
    ys: Vector[Data],
    theta: Vector[(Double, DenseVector[Double])]): Rand[DenseMatrix[Double]] = {

    val ssy = (theta zip ys)
      .map {
        case ((time, x), y) =>
          val fm = KalmanFilter.missingF(f, time, y.observation)
          val yt = KalmanFilter.flattenObs(y.observation)
          (yt - fm.t * x) *:* (yt - fm.t * x)
      }
      .reduce(_ + _)

    val shape = prior.shape + ys.size * 0.5
    val rate = ssy.map(ss => prior.scale + ss * 0.5)

    val res = rate.map(r => InverseGamma(shape, r).draw)

    Rand.always(diag(res))
  }

  /**
    * Sample the diagonal system matrix for an irregularly observed
    * DLM
    */
  def sampleSystemMatrix(
    prior: InverseGamma,
    theta: Vector[(Double, DenseVector[Double])],
    g: Double => DenseMatrix[Double]): Rand[DenseMatrix[Double]] = {

    // take the squared difference of x_t - g * x_{t-1} for t = 1 ... 0
    // add them all up
    val squaredSum = theta.init
      .zip(theta.tail)
      .map {
        case (mt, mt1) =>
          val dt = mt1._1 - mt._1
          val diff = (mt1._2 - g(dt) * mt._2)
          (diff *:* diff) / dt
      }
      .reduce(_ + _)

    val shape = prior.shape + (theta.size - 1) * 0.5
    val rate = squaredSum map (s => prior.scale + s * 0.5)

    val res = rate.map(r => InverseGamma(shape, r).draw)

    Rand.always(diag(res))
  }

  /**
    * Calculate the marginal likelihood of phi given
    * the values of the latent-state and other static parameters
    * @param state a sample of the latent state of an AR(1) DLM
    * @param p the static parameters of a DLM
    * @param phi autoregressive
    */
  def arlikelihood(state: Vector[(Double, DenseVector[Double])],
                   p: DlmParameters,
                   phi: DenseVector[Double]): Double = {

    val n = state.length
    val ssa = (state.init zip state.tail)
      .map { case (at, at1) => at1._2 - phi * at._2 }
      .map(x => x * x)
      .reduce(_ + _)

    val det = sum(log(diag(cholesky(p.w))))

    -n * 0.5 * log(2 * math.Pi) + det - 0.5 * (p.w \ ssa).dot(ssa)
  }

  /**
    * Update an autoregressive model with a new value of the autoregressive
    * parameter
    */
  def updateModel(mod: DlmModel, phi: Double*): DlmModel = {

    mod.copy(g = (dt: Double) => new DenseMatrix(phi.size, 1, phi.toArray))
  }

  /**
    * Sample the autoregressive parameter with a Beta Prior
    * and proposal distribution
    * @param
    */
  def samplePhi(prior: Beta, lambda: Double, tau: Double, s: State) = {

    val proposal = (phi: Double) =>
      new Beta(lambda * phi + tau, lambda * (1 - phi) + tau)

    val pos = (phi: Double) =>
      prior.logPdf(phi) + arlikelihood(s.state, s.p, DenseVector(phi))

    MarkovChain.Kernels.metropolisHastings(proposal)(pos)
  }

  /**
    * A single step of a Gibbs Sampler
    * @param mod the model containing the definition of
    * the observation matrix F_t and system evolution matrix G_t
    * @param priorV the prior distribution on the observation noise matrix, V
    * @param priorW the prior distribution on the system noise matrix, W
    * @param observations an array of Data containing the observed time series
    */
  def dinvGammaStep(
    mod: DlmModel,
    priorV: InverseGamma,
    priorW: InverseGamma,
    observations: Vector[Data]) = { s: State =>
    
    for {
      theta <- Smoothing.ffbs(mod, observations,
        KalmanFilter.advanceState(s.p, mod.g), Smoothing.step(mod, s.p.w), s.p)
      newV <- sampleObservationMatrix(priorV,
                                      mod.f,
                                      observations,
                                      theta.toVector)
      newW <- sampleSystemMatrix(priorW, theta.toVector, mod.g)
    } yield State(s.p.copy(v = newV, w = newW), theta.toVector)
  }

  /**
    * Return a Markov chain using Gibbs Sampling to determine
    * the values of the system and
    * observation noise covariance matrices, W and V
    * @param mod the model containing the definition of the observation
    * matrix F_t and system evolution matrix G_t
    * @param priorV the prior distribution on the observation noise matrix, V
    * @param priorW the prior distribution on the system noise matrix, W
    * @param initParams the initial parameters of the Markov Chain
    * @param observations an array of Data containing the observed time series
    * @return a Process
    */
  def sample(
    mod: DlmModel,
    priorV: InverseGamma,
    priorW: InverseGamma,
    initParams: DlmParameters,
    observations: Vector[Data]) = {

    val init = for {
      initState <- Smoothing.ffbs(mod, observations,
        KalmanFilter.advanceState(initParams, mod.g),
        Smoothing.step(mod, initParams.w), initParams)
    } yield State(initParams, initState)

    MarkovChain(init.draw)(dinvGammaStep(mod, priorV, priorW, observations))
  }

  def stepSvd(
    mod: DlmModel,
    priorV: InverseGamma,
    priorW: InverseGamma,
    observations: Vector[Data]) = { s: State =>
    
    for {
      theta <- SvdSampler.ffbs(mod, observations, s.p, SvdFilter.advanceState(s.p, mod.g))
      newV <- sampleObservationMatrix(priorV,
                                      mod.f,
                                      observations,
                                      theta.toVector)
      newW <- sampleSystemMatrix(priorW, theta.toVector, mod.g)
    } yield State(s.p.copy(v = newV, w = newW), theta.toVector)
  }

  /**
    * Perform Gibbs Sampling using the SVD Kalman Filter for numerical stability
    */
  def sampleSvd(
    mod: DlmModel,
    priorV: InverseGamma,
    priorW: InverseGamma,
    initParams: DlmParameters,
    observations: Vector[Data]) = {

    val init = for {
      initState <- SvdSampler.ffbs(mod, observations, initParams,
        SvdFilter.advanceState(initParams, mod.g))
    } yield State(initParams, initState)

    MarkovChain(init.draw)(stepSvd(mod, priorV, priorW, observations))
  }

  /**
    * Calculate the marginal likelihood for metropolis hastings
    */
  def likelihood(
    theta: Vector[(Double, DenseVector[Double])],
    g: Double => DenseMatrix[Double])(p: DlmParameters): Double = {

    val n = theta.length
    val ssa = (theta.init zip theta.tail)
      .map {
        case (at, at1) =>
          val dt = at1._1 - at._1
          at1._2 - g(dt) * at._2
      }
      .map(x => x * x)
      .reduce(_ + _)

    val det: Double = sum(log(diag(cholesky(p.w))))
    -n * 0.5 * log(2 * math.Pi) + det - 0.5 * (p.w \ ssa).dot(ssa)
  }

  /**
    * A metropolis step for a DLM
    * @param mod a DLM model
    * @param theta the currently sampled state of the DLM
    * @param proposal a symmetric proposal distribution for the parameters
    * of a DLM
    * @return a function from Parameters => Rand[Parameters] which
    * performs a metropolis step to be used in a Markov Chain
    */
  def metropStep(
    mod: DlmModel,
    theta: Vector[(Double, DenseVector[Double])],
    proposal: DlmParameters => Rand[DlmParameters]) = {

    MarkovChain.Kernels.metropolis(proposal)(likelihood(theta, mod.g))
  }

  /**
    * Use metropolis hastings to determine the initial state
    * distribution x0 ~ N(m0, C0)
    * @param proposal a proposal distribution for the parameters of the initial state
    * @param mod a DLM model specification
    * @param priorV the prior distribution of the observation noise matrix
    * @param priorW the prior distribution of the system noise matrix
    * @param observations a vector of observations
    * @param gibbsState the current state of the Markov Chain
    */
  def gibbsMetropStep(
    proposal: DlmParameters => Rand[DlmParameters],
    mod: DlmModel,
    priorV: InverseGamma,
    priorW: InverseGamma,
    observations: Vector[Data]) = { s: State =>

    for {
      theta <- SvdSampler.ffbs(mod, observations, s.p,
        SvdFilter.advanceState(s.p, mod.g))
      thetaSorted = theta.sortBy(_._1)
      newV <- sampleObservationMatrix(priorV, mod.f, observations, thetaSorted)
      newW <- sampleSystemMatrix(priorW, thetaSorted, mod.g)
      newP <- metropStep(mod, thetaSorted, proposal)(s.p)
    } yield State(newP.copy(v = newV, w = newW), thetaSorted)
  }

  /**
    *
    */
  def metropSamples(
    proposal: DlmParameters => Rand[DlmParameters],
    mod: DlmModel,
    priorV: InverseGamma,
    priorW: InverseGamma,
    initParams: DlmParameters,
    observations: Vector[Data]) = {

    val init = for {
      initState <- Smoothing.ffbs(mod, observations,
        KalmanFilter.advanceState(initParams, mod.g),
        Smoothing.step(mod, initParams.w), initParams)
    } yield State(initParams, initState)

    MarkovChain(init.draw)(
      gibbsMetropStep(proposal, mod, priorV, priorW, observations))
  }
}
