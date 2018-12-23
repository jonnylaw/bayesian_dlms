package dlm.core.model

import cats.implicits._
import breeze.linalg.{DenseMatrix, diag, sum}
import breeze.stats.distributions._

/**
  * Fit a state space model with a latent Gaussian state and a Student's t observation distribution
  * using the fact that the student's t-distribution is an Inverse Gamma mixture of normals
  */
object StudentT {

  /**
    * The state of the Markov chain for the Student's t-distribution
    * gibbs sampler
    * @param p the DLM parameters
    * @param variances the variances for each of the observations V_t
    * @param nu the degrees of freedom of the Student's t-distribution
    * @param state the currently sampled state using FFBS
    */
  case class State(p: DlmParameters,
                   variances: Vector[Double],
                   nu: Int,
                   state: Vector[SamplingState],
                   accepted: Int)

  /**
    * Sample the variances of the Normal distribution
    * These are auxilliary variables required when calculating
    * the one-step prediction in the Kalman Filter
    * @param ys an array of observations of length N
    * @param f the observation matrix
    * @param dof the degrees of freedom of the Student's t-distribution
    * @return a Rand distribution over the list of N variances
    */
  def sampleVariances(ys: Vector[Data],
                      f: Double => DenseMatrix[Double],
                      dof: Int,
                      theta: Vector[SamplingState],
                      p: DlmParameters) = {

    val scale = p.v(0, 0)
    val alpha = (dof + 1) * 0.5

    val diff = (ys.map(_.observation) zip theta)
      .map {
        case (y, s) =>
          val ft = f(s.time).t * s.sample
          y(0).map(y => (y - ft(0)) * (y - ft(0))).getOrElse(0.0)
      }

    val beta = diff.map(d => (dof * scale * 0.5) + d * 0.5)

    beta map (b => InverseGamma(alpha, b).draw)
  }

  /**
    * Calculate the log-likelihood of the student's t-distributed model
    * @param ys a vector of observations
    * @param xs a sample from the latent-state of the student's t model
    * @param mod the Student t DGLM
    * @param index the index of the observation to select
    * @param p the
    */
  def ll(ys: Vector[Data],
         xs: Vector[SamplingState],
         mod: Dglm,
         p: DlmParameters)(nu: Int) = {

    (xs.tail zip ys)
      .map {
        case (s, Data(t, yo)) =>
          yo(0)
            .map { y =>
              val location = mod.f(t).t * s.sample
              val scale = math.sqrt(p.v(0, 0))
              ScaledStudentsT(nu, location(0), scale).logPdf(y)
            }
            .getOrElse(0.0)
      }
      .reduce(_ + _)
  }

  /**
    * Sample the (square of the) scale of the Student's t distribution
    * @param dof the degrees of freedom of the Student's t observation distribution
    * @param vs the current sampled scales
    */
  def sampleScaleT(dof: Int, variances: Vector[Double]) = {

    val t = variances.size

    val shape = t * dof * 0.5 + 1
    val rate = dof * 0.5 * variances.map(1.0 / _).sum
    val scale = 1 / rate

    Gamma(shape, scale)
  }

  def filter(mod: Dlm,
             variances: Vector[Double],
             observations: Vector[Data],
             params: DlmParameters) = {

    // create a list of parameters with the variance in them
    val ps = variances.map(vi => params.copy(v = DenseMatrix(vi)))

    val kf = (p: DlmParameters) =>
      KalmanFilter(KalmanFilter.advanceState(p, mod.g))
    def kalmanStep(p: DlmParameters) = kf(p).step(mod, p) _

    val (at, rt) =
      KalmanFilter.advState(mod.g, params.m0, params.c0, 0, params.w)
    val init = kf(params).initialiseState(mod, params, observations)

    // fold over the list of variances and the observations
    (ps zip observations).scanLeft(init) {
      case (s, (p, y)) => kalmanStep(p)(s, y)
    }
  }

  /**
    * Sample the state, incorporating the drawn variances for each observation
    * @param variances the sampled auxiliary parameters
    * @param mod the DLM
    * @param observations
    * @param params the parameters of the DLM model
    */
  def sampleState(variances: Vector[Double],
                  mod: Dlm,
                  observations: Vector[Data],
                  params: DlmParameters) = {

    val filtered = filter(mod, variances, observations, params)
    Rand.always(Smoothing.sampleDlm(mod, filtered, params.w))
  }

  /**
    * Sample the degrees of freedom for the observation distribution
    */
  def sampleNu(prop: Int => Rand[Int],
               propP: (Int, Int) => Double,
               prior: Int => Double,
               likelihood: Int => Double) = { (nu: Int) =>
    val logMeasure = (nu: Int) => likelihood(nu) + prior(nu)

    for {
      propNu <- prop(nu)
      a = logMeasure(propNu) + propP(propNu, nu) -
        logMeasure(nu) - propP(nu, propNu)
      u <- Uniform(0, 1)
      next = if (math.log(u) < a) (propNu, true) else (nu, false)
    } yield next
  }

  /**
    * Sample the latent state for the student's t model
    */
  def interpolateStep(data: Vector[Data], mod: Dglm, p: DlmParameters) = {
    s: State =>
      val dlm = Dlm(mod.f, mod.g)

      for {
        theta <- sampleState(s.variances, dlm, data, p)
        vs = sampleVariances(data, mod.f, s.nu, theta, p)
      } yield State(s.p, vs, s.nu, theta, s.accepted)
  }

  def interpolate(data: Vector[Data], mod: Dglm, nu: Int, p: DlmParameters) = {

    val dlm = Dlm(mod.f, mod.g)
    val initVariances = Vector.fill(data.size)(1.0)
    val initState = sampleState(initVariances, dlm, data, p)
    val init = State(p, initVariances, nu, initState.draw, 0)

    MarkovChain(init)(interpolateStep(data, mod, p))
  }

  /**
    * A single step of the Student t-distribution Gibbs Sampler
    */
  def step(data: Vector[Data],
           priorW: InverseGamma,
           priorNu: DiscreteDistr[Int],
           propNu: Int => Rand[Int],
           propNuP: (Int, Int) => Double,
           mod: Dglm,
           p: DlmParameters) = { s: State =>
    val dlm = Dlm(mod.f, mod.g)

    for {
      theta <- sampleState(s.variances, dlm, data, p)
      state = theta.map(s => (s.time, s.sample))
      newW <- GibbsSampling.sampleSystemMatrix(priorW, state, mod.g)
      vs = sampleVariances(data, mod.f, s.nu, theta, p)
      scale <- sampleScaleT(s.nu, vs)
      (nu, accepted) <- sampleNu(propNu,
                                 propNuP,
                                 priorNu.logProbabilityOf,
                                 ll(data, theta, mod, p))(s.nu)
    } yield
      State(s.p.copy(v = DenseMatrix(scale), w = newW),
            vs,
            nu,
            theta,
            s.accepted + (if (accepted) 1 else 0))
  }

  /**
    * Perform Gibbs Sampling for the Student t-distributed model
    * @param priorW the prior distribution of the system noise matrix
    * @param mod the DGLM representing the Student's t model
    * @param params the initial parameters
    */
  def sample(data: Vector[Data],
             priorW: InverseGamma,
             priorNu: DiscreteDistr[Int],
             propNu: Int => Rand[Int],
             propNuP: (Int, Int) => Double,
             mod: Dglm,
             params: DlmParameters) = {

    val dlm = Dlm(mod.f, mod.g)
    val initVariances = Vector.fill(data.size)(1.0)
    val initState = sampleState(initVariances, dlm, data, params)
    val init = State(params, initVariances, priorNu.draw, initState.draw, 0)

    MarkovChain(init)(step(data, priorW, priorNu, propNu, propNuP, mod, params))
  }

  case class PmmhState(ll: Double, p: DlmParameters, nu: Int, accepted: Int)

  /**
    * Perform a single step of the PMMH algorithm for the Student's t distributed state
    * space model
    * @param priorW
    */
  def stepPmmh(priorW: ContinuousDistr[Double],
               priorV: ContinuousDistr[Double],
               priorNu: DiscreteDistr[Int],
               prop: DlmParameters => Rand[DlmParameters],
               propNu: Int => Rand[Int],
               propNuP: (Int, Int) => Double,
               ll: (DlmParameters, Int) => Double) = { s: PmmhState =>
    val logMeasure = (p: DlmParameters, nu: Int) =>
      ll(p, nu) +
        priorNu.logProbabilityOf(nu) +
        sum(diag(p.w).map(wi => priorW.logPdf(wi))) +
        sum(diag(p.v).map(vi => priorV.logPdf(vi)))

    for {
      nu <- propNu(s.nu)
      propP <- prop(s.p)
      ll = logMeasure(propP, nu)
      a = ll + propNuP(nu, s.nu) - s.ll - propNuP(s.nu, nu)
      u <- Uniform(0, 1)
      next = if (math.log(u) < a) {
        PmmhState(ll, propP, nu, s.accepted + 1)
      } else {
        s
      }
    } yield next
  }

  /**
    * Particle Marginal Metropolis Hastings for the Student's t-distributed state space model
    */
  def samplePmmh(data: Vector[Data],
                 priorW: ContinuousDistr[Double],
                 priorV: ContinuousDistr[Double],
                 priorNu: DiscreteDistr[Int],
                 prop: DlmParameters => Rand[DlmParameters],
                 propNu: Int => Rand[Int],
                 propNuP: (Int, Int) => Double,
                 model: Dlm,
                 n: Int,
                 initP: DlmParameters,
                 initNu: Int): Process[PmmhState] = {

    val mod = (nu: Int) => Dglm.studentT(nu, model)
    val ll = (p: DlmParameters, nu: Int) =>
      ParticleFilter.likelihood(mod(nu), data, n)(p)

    val init = PmmhState(-1e99, initP, initNu, 0)
    MarkovChain(init)(
      stepPmmh(priorW, priorV, priorNu, prop, propNu, propNuP, ll))
  }
}
