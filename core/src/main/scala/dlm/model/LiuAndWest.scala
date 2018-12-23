package dlm.core.model

// exclude vector
import breeze.linalg.{Vector => _, _}
import breeze.stats.distributions._
import cats.Traverse
import cats.implicits._
import math.{exp, sqrt, log}
import spire.implicits._

/**
  * State for the particle filter with parameters
  * @param time the time of the observation
  * @param state a particle cloud representing the latent-state
  * @param weights the conditional log-likelihood of the latent-state
  * @param params a particle cloud representing the values of the parameters
  */
case class PfStateParams(time: Double,
                         state: Vector[DenseVector[Double]],
                         weights: Vector[Double],
                         params: Vector[DlmParameters])

/**
  * Extended Particle filter which approximates the parameters as a particle cloud
  */
case class LiuAndWestFilter(n: Int,
                            prior: Rand[DlmParameters],
                            a: Double,
                            n0: Int)
    extends FilterTs[PfStateParams, DlmParameters, Dglm] {
  import LiuAndWestFilter._

  def initialiseState[T[_]: Traverse](
      model: Dglm,
      p: DlmParameters,
      ys: T[Data]
  ): PfStateParams = {

    val t0 = ys.map(_.time).reduceLeftOption((t0, d) => math.min(t0, d))
    val x0 = MultivariateGaussian(p.m0, p.c0).sample(n)
    val p0 = prior.sample(n).map(_.map(log))
    val w0 = Vector.fill(n)(1.0)

    PfStateParams(t0.get - 1.0, x0.toVector, w0, p0.toVector)
  }

  def step(mod: Dglm, p: DlmParameters)(x: PfStateParams,
                                        d: Data): PfStateParams = {

    val meanParams = weightedMeanParams(x.params, x.weights)
    val varParams = varParameters(x.params)
    val mi = scaleParameters(x.params, meanParams, a)
    val dt = d.time - x.time

    val y = KalmanFilter.flattenObs(d.observation)
    val thetaHat = (mi zip x.state).map { case (m, t) =>
      Dglm.stepState(mod, m map exp)(t, dt).mean }
    val auxVars = auxiliaryVariables(x.weights, thetaHat, mod, y, mi)

    val (newParams, newState, logw) = (for {
      i <- auxVars
      param = proposal(mi(i), diag(varParams * (1 - a * a)))
      newState = Dglm.stepState(mod, param map exp)(x.state(i), dt).draw
      topw = mod.conditionalLikelihood(param.v map exp)(newState, y)
      bottomw = mod.conditionalLikelihood(mi(i).v map exp)(thetaHat(i), y)
    } yield (param, newState, topw - bottomw)).unzip3

    val maxWeight = logw.max
    val ws = logw map (w => exp(w - maxWeight))
    val ess =
      ParticleFilter.effectiveSampleSize(ParticleFilter.normaliseWeights(ws))

    // if the effective sample size is less than a n0, then resample
    if (ess < n0) {
      val (np, ns) =
        ParticleFilter.multinomialResample(newParams zip newState, ws).unzip
      PfStateParams(d.time, ns, Vector.fill(n)(1.0), np)
    } else {
      PfStateParams(d.time, newState, logw, newParams)
    }
  }
}

object LiuAndWestFilter {

  /**
    * Calculate the auxiliary variables needed for online importance sampling
    */
  def auxiliaryVariables(weights: Vector[Double],
                         states: Vector[DenseVector[Double]],
                         mod: Dglm,
                         y: DenseVector[Double],
                         mi: Vector[DlmParameters]): Vector[Int] = {

    val logAuxWeights = (weights, mi, states).zipped map {
      case (weight, param, state) =>
        val ll = mod.conditionalLikelihood(param.v map exp)(state, y)
        weight + ll
    }

    val max = logAuxWeights.max
    val auxWeights = logAuxWeights map (a => exp(a - max))

    ParticleFilter.multinomialResample(states.indices.toVector, auxWeights)
  }

  def scaleParameters(params: Vector[DlmParameters],
                      meanParams: DlmParameters,
                      a: Double): Vector[DlmParameters] = {

    for {
      param <- params
      mp = meanParams.map(x => x * (1.0 - a))
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
  def advanceState(p: DlmParameters, model: Dglm)(s: PfStateParams,
                                                  dt: Double) = {

    val adv = s.state map (x => Dglm.stepState(model, p)(x, dt).draw)
    s.copy(state = adv)
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
  def credibleIntervals(xs: Vector[DenseVector[Double]],
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
    * Calculate the weighted mean of v and w parameter particles
    */
  def weightedMeanParams(
    ps: Vector[DlmParameters],
    logws: Vector[Double]): DlmParameters = {

    val ws = normalise(logws)

    ps.zip(ws).
      map { case (p, w) => p.map(_ * w) }.
      reduce(_ add _)
  }

  def normalise(logws: Vector[Double]): Vector[Double] = {
    val maximum = logws.max
    val ws = logws map (w => exp(w - maximum))
    ParticleFilter.normaliseWeights(ws)
  }

  /**
    * Calculate the weighted variance
    */
  def weightedMeanVarianceParams(
    ps: Vector[DlmParameters],
    logws: Vector[Double]): (DlmParameters, DlmParameters) = {
    val wm = weightedMeanParams(ps, logws)
    val diff = ps.map { p =>
      val c = p minus wm
      c times c
    }
    (wm, weightedMeanParams(diff, logws))
  }

  /**
    * Calculate the variance of the parameter particles
    */
  def varParameters(p: Vector[DlmParameters]): DenseVector[Double] = {
    val m = seqToMatrix(p map (x => DenseVector(x.toList.toArray)))
    breeze.stats.variance(m(::, *)).t
  }

  def weightedVarParameters(
    ps: Vector[DlmParameters],
    ws: Vector[Double]): DenseVector[Double] = {
    val m = seqToMatrix((ps zip ws).map { case (p, w) =>
                          val wp = p.map(_ * w)
                          DenseVector(wp.toList.toArray)
                        })
    breeze.stats.variance(m(::, *)).t
  }

  /**
    * Create a matrix from v and w
    */
  def paramsToMatrix(p: DlmParameters): DenseMatrix[Double] = {
    Dlm.blockDiagonal(p.v, p.w)
  }

  /**
    * Multivariate Normal proposal distribution for the logged-parameters psi = (log(v), log(w))
    * @param p the DLM parameters
    * @param delta the covariance matrix of the MVN proposal distribution
    * @return perturbed logged DLM parameters
    */
  def proposal(p: DlmParameters, delta: DenseMatrix[Double]) = {
    val z = DenseVector.rand(delta.cols, Gaussian(0.0, 1.0))
    val innov = delta.map(sqrt) * z
    val newV = diag(p.v) + innov(0 until p.v.cols)
    val newW = diag(p.w) + innov(p.v.cols until (p.v.cols + p.w.cols))

    p.copy(v = diag(newV), w = diag(newW))
  }
}
