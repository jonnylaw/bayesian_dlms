package dlm.model

import Dglm._
import ParticleGibbs.State
import breeze.stats.distributions.{Multinomial, Rand}
import breeze.linalg.DenseVector
import breeze.stats.mean
import math.{log, exp}
import cats.implicits._

/**
  * Particle Gibbs with Ancestor Sampling
  * Requires a Tractable state evolution kernel
  */
object ParticleGibbsAncestor extends App {

  /**
    * Calculate the importance weights for ancestor resampling
    * the Nth path 
    */
  def importanceWeight(
    sampledStates:    Vector[DenseVector[Double]],
    weights:          Vector[Double],
    conditionedState: DenseVector[Double],
    time:             Time,
    mod:              Model,
    p:                Dlm.Parameters
  ): Vector[Double] = {

    val res = for {
      (x, w) <- (sampledStates, weights).zipped
      ll = transitionProbability(x, time, mod, p) _
    } yield w * ll(conditionedState)

    res.toVector
  }

  /** 
    * Calculate the transition probability from a given particle at time t-1
    * to the conditioned upon particle at time t
    * @param sampledState a particle value at time t-1
    * @param time the time t
    * @param mod the DGLM model specification
    * @param p the parameters of the model
    * @param conditionedState the value of the conditioned state at time t
    * @return the value of the log of the transition probability as a Double
    */
  def transitionProbability(
    sampledState: DenseVector[Double],
    time:         Time,
    mod:          Model,
    p:            Dlm.Parameters)(conditionedState: DenseVector[Double]) = {

    MultivariateGaussianSvd(mod.g(time) * sampledState, p.w).
      logPdf(conditionedState)
  }
  /**
    * Sample n items with replacement from xs with probability ws
    */
  def sample[A](n: Int, xs: Vector[A], ws: Vector[Double]): Vector[A] = {
    val indices = Multinomial(DenseVector(ws.toArray)).sample(n)

    indices.map(xs(_)).toVector
  }

  /**
    * Recalculate the weights such that the smallest weight is 1
    */
  def logSumExp(w: Vector[Double]) = {
    val largest = w.max
    w map (a => exp(a - largest))
  }

  /**
    * Perform Ancestor Resampling on Particle Paths
    * @param mod a DGLM model
    * @param time the time of the observation
    * @param states a vector containing the latent states up to the current time 
    * @param weights a vector of weights calculated at the previous time step
    * @param condState the previously conditioned upon path
    * @param p the parameters of the model, including W
    * @return a tuple where the first element is a list describing the paths from 0:time-1, 
    * and second element describes the particles at the current time
    */
  def ancestorResampling(
    mod:       Model,
    time:      Time,
    states:    Vector[LatentState],
    weights:   Vector[Double],
    condState: DenseVector[Double],
    p:         Dlm.Parameters): (List[LatentState], LatentState) = {
    val n = states.head.size

    // sample n-1 particles from ALL of the states
    val x = sample(n-1, states.transpose, weights).map(_.toList).toList.transpose

    // calculate transition probability of each particle to the next conditioned state
    val transLogProb = importanceWeight(states.head.map(_._2).toVector, weights, 
      condState, time, mod, p)

    val transProb = logSumExp(transLogProb)

    // sample the nth path x_{1:t-1}^N proportional to transProb
    val xn = sample(1, states.transpose, transProb).head.toList

    // advance the n-1 states
    val x1 = ParticleFilter.advanceState(mod.g, time, x.head.map(_._2), p).draw.toList

    // set the nth particle at time t to the conditioned state at time t
    // this is the state at time t
    val xt = (condState :: x1).map((time, _))

    // return a tuple containing a list describing the paths from 0:t-1, and the particles at time t
    ((xn :: x.transpose).transpose, xt)
  }

/**
  * The weights are proportional to the conditional likelihood of the observations
  * given the state multiplied by the transition probability of the resampled 
  * particles at time t-1 to the conditioned state at time t
  * TODO: This could be wrong, as we do not calculate the transition probability properly
  * @param mod the DGLM model
  * @param time the current time, t
  * @param xt the value of the state at time t
  * @param xt1 the value of the state at time t-1
  * @param conditionedState the value of the conditioned state at time t
  * @param y the observation at time t
  * @param p the parameters of the model
  * @return the conditional likeihood of the observation given the state and the
  * transition probability to the next conditioned upon state
  */
  def calcWeight(
    mod:              Model,
    time:             Time,
    xt:               DenseVector[Double],
    xt1:              DenseVector[Double],
    conditionedState: DenseVector[Double],
    y:                Observation,
    p:                Dlm.Parameters): Double = {

    mod.conditionalLikelihood(p)(y, xt) + 
    transitionProbability(xt1, time, mod, p)(conditionedState)
  }

  /**
    * A single step in the Particle Gibbs with Ancestor Sampling algorithm
    * @param mod the model specification
    * @param p the parameters of the model
    * @return the state containing the particles, weights and running log-likelihood
    */
  def step(
    mod: Model,
    p:   Dlm.Parameters
  ) = (s: State, a: (Data, DenseVector[Double])) => a match {

    case (Data(time, Some(y)), conditionedState) =>
      val (prevStates, statet) = ancestorResampling(mod, time, 
        s.states.toVector, s.weights.toVector, 
        conditionedState, p)

      // calculate the weights
      val w = (prevStates.head, statet).
        zipped.
        map { case (xt1, xt) => 
          calcWeight(mod, time, xt._2, xt1._2, conditionedState, y, p) 
        }

      // log-sum-exp and calculate log-likelihood
      val max = w.max
      val w1 = w map (a => exp(a - max))
      val ll = s.ll + max + log(mean(w1))

      State(statet :: prevStates, w1, ll)

    case (Data(time, None), conditionedState) => 
      val n = s.states.size
      val (xt1, xt) = ancestorResampling(mod, time, 
        s.states.toVector, s.weights.toVector, conditionedState, p)

      State(xt :: xt1, List.fill(n)(1.0 / n), s.ll)
  }

  /**
    * Run Particle Gibbs with Ancestor Sampling
    * @param n the number of particles to have in the sampler
    * @param mod the DGLM model specification
    * @param p the model parameters
    * @param obs a list of measurements
    * @param state the state which is to be conditioned upon
    * @return the state of the Particle Filter, including all Paths
    */
  def filterAll(
    n:   Int,
    mod: Model,
    p:   Dlm.Parameters,
    obs: List[Data])(state: LatentState): State = {

    val firstTime = obs.map(d => d.time).min
    val x0 = ParticleGibbs.initState(p).
      sample(n-1).
      toList.
      map(x => (firstTime - 1, x))
    val init = State(List(state.head :: x0), List.fill(n)(1.0 / n), 0.0)

    (obs, state.map(_._2)).
      zipped.
      foldLeft(init)(step(mod, p))
  }

  /**
    * Run Particle Gibbs with Ancestor Sampling
    * @param n the number of particles to have in the sampler
    * @param mod the DGLM model specification
    * @param p the model parameters
    * @param obs a list of measurements
    * @param state the state which is to be conditioned upon
    * @return a distribution over a tuple containing the log-likelihood and 
    * sampled state from the PGAS kernel
    */
  def filter(
    n:   Int,
    p:   Dlm.Parameters,
    mod: Model,
    obs: List[Data])(state: LatentState): Rand[(Double, LatentState)] = {
    val filtered = filterAll(n, mod, p, obs)(state)

    ParticleGibbs.sampleState(filtered.states, filtered.weights).
      map((filtered.ll, _))
  }
}
