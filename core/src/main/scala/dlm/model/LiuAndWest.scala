package core.dlm.model

// exclude vector
import breeze.linalg.{Vector => _, _}
import breeze.stats.distributions._
import cats.Traverse
import cats.implicits._
import math.{exp, sqrt, log}

/**
  * State for the conjugate filter
  * @param time the time of the observation
  * @param state a particle cloud representing the latent-state
  * @param weights the conditional log-likelihood of the latent-state 
  * @param params a particle cloud representing the values of the parameters
  */
case class PfStateParams(
  time:      Double,
  state:     Vector[DenseVector[Double]],
  weights:   Vector[Double],
  params:    Vector[DlmParameters])

/**
  * Extended Particle filter which approximates the parameters as a particle cloud
  */
case class LiuAndWestFilter(n: Int, prior: Rand[DlmParameters], a: Double)
    extends FilterTs[PfStateParams, DlmParameters, DglmModel] {
  import LiuAndWestFilter._

  def initialiseState[T[_]: Traverse](
    model: DglmModel,
    p: DlmParameters,
    ys: T[Dlm.Data]
  ): PfStateParams = {

    val t0 = ys.map(_.time).reduceLeftOption((t0, d) => math.min(t0, d))
    val x0 = MultivariateGaussian(p.m0, p.c0).sample(n)
    val p0 = prior.sample(n).map(_.map(log))
    val w0 = Vector.fill(n)(1.0 / n)
    PfStateParams(t0.get - 1.0, x0.toVector, w0, p0.toVector)
  }

  def step(
    mod: DglmModel,
    p:   DlmParameters)
    (x:  PfStateParams, d: Dlm.Data): PfStateParams = {

    val varParams = varParameters(x.params)
    val mi = scaleParameters(x.params, a)

    val y = KalmanFilter.flattenObs(d.observation)
    val auxVars = auxiliaryVariables(x.weights, x.state, mod, y, mi)

    // propose new log-parameters
    val newParams = for {
      i <- auxVars
      param = mi(i)
      p = proposal(param, diag(varParams * (1 - a * a)))
    } yield p

    val dt = d.time - x.time
    val newState: Vector[DenseVector[Double]] = for {
      i <- auxVars
      param = newParams(i)
      state = x.state(i)
    } yield Dglm.stepState(mod, param map exp, state, dt).draw

    val topw = (newState zip newParams) map { case (state, param) =>
      mod.conditionalLikelihood(param.v map exp)(state, y)
    }

    val bottomw = for {
      i <- auxVars
      state = x.state(i)
      param = mi(i)
    } yield mod.conditionalLikelihood(param.v map exp)(state, y)

    val logw = (topw zip bottomw).map {
      case (tw, bw) => tw - bw }

    val max1 = logw.max
    val w = logw map (a => exp(a - max1))

    PfStateParams(d.time, newState, w, newParams)
  }
}

object LiuAndWestFilter {

  /**
    * Calculate the auxiliary variables needed for online importance sampling
    */
  def auxiliaryVariables(
    weights: Vector[Double],
    states: Vector[DenseVector[Double]],
    mod: DglmModel,
    y:   DenseVector[Double],
    mi:  Vector[DlmParameters]): Vector[Int] = {

    val logAuxWeights = (weights, mi, states).zipped map { case (weight, param, state) =>
      val ll = mod.conditionalLikelihood(param.v map exp)(state, y)
      weight + ll
    }

    val max = logAuxWeights.max
    val auxWeights = logAuxWeights map (a => exp(a - max))

    ParticleFilter.multinomialResample(states.indices.toVector, auxWeights)
  }

  def scaleParameters(
    params: Vector[DlmParameters],
    a: Double): Vector[DlmParameters] = {
    val meanParams: DlmParameters = meanParameters(params)

    for {
      param <- params
      mp = meanParams.map(x => x * (1.0-a))
      m = param.map(_ * a).add(mp)
    } yield m
  }

  /**
    * Advance the state particles to the a-priori
    * value of the state at time t
    * @param s the current state of the Kalman Filter
    * @param dt the time increment
    * @return the a-priori mean and covariance of the state at time t
    */
  def advanceState(
    p: DlmParameters,
    model: DglmModel)
    (s:  PfStateParams, dt: Double) = {

    val adv = s.state traverse (x => Dglm.stepState(model, p, x, dt))
    s.copy(state = adv.draw)
  }

  /**
    * Function to create a DenseMatrix from a sequence of DenseVectors
    * @param v a sequence of DenseVectors
    * @return a denseMatrix with columns corresponding to the elements of v
    */
  def seqToMatrix(v: Vector[DenseVector[Double]]): DenseMatrix[Double] = {
    val elems = v.reduce((a, b) => DenseVector.vertcat(a, b))

    new DenseMatrix(v.head.size, v.length, elems.data).t
  }

  def meanState(v: Vector[DenseVector[Double]]): DenseVector[Double] =
    v.reduce(_ + _).map(_ / v.size)

  def varState(v: Vector[DenseVector[Double]]): DenseVector[Double] = {
    val m = seqToMatrix(v)
    breeze.stats.variance(m(::, *)).t
  }


  /**
    * Determine the credible intervals of a collection of samples of DenseVectors
    * @param
    */
  def credibleIntervals(
    xs:       Vector[DenseVector[Double]],
    interval: Double): Vector[(Double, Double)] = {

    for {
      x <- xs.map(_.data.toVector).transpose
      n = xs.size
      upper = math.floor(n * interval).toInt
      lower = n - upper
    } yield (x(lower), x(upper))
  }

  /**
    * Calculate the mean of the parameter particles
    */
  def meanParameters(p: Vector[DlmParameters]): DlmParameters = {
    p.reduce(_ add _).map(_ / p.size)
  }

  /**
    * Calculate the variance of the parameter particles 
    */
  def varParameters(p: Vector[DlmParameters]): DenseVector[Double] = {
    val m = seqToMatrix(p map (x => DenseVector(x.toList.toArray)))
    breeze.stats.variance(m(::, *)).t
  }

  /**
    * Multivariate Normal proposal distribution for the logged-parameters psi = (log(v), log(w))
    * @param p the DLM parameters
    * @param delta the covariance matrix of the MVN proposal distribution
    * @return perturbed logged DLM parameters 
    */
  def proposal(p: DlmParameters, delta: DenseMatrix[Double]) = {
    val innov = delta.map(sqrt) * DenseVector(Gaussian(0.0, 1.0).sample(delta.cols).toArray)
    val newV = diag(p.v) + innov(0 until p.v.cols)
    val newW = diag(p.w) + innov(p.v.cols until (p.v.cols + p.w.cols))

    p.copy(v = diag(newV), w = diag(newW))
  }
}
