 package dlm.core.model

import breeze.stats.distributions._
import breeze.linalg.DenseVector
import math._

case class StochVolState(
  params:   SvParameters,
  alphas:   Vector[FilterAr.SampleState])

/**
  * Use a Gaussian approximation to the state space to sample the
  * stochastic volatility model with discrete regular observations
  * and an AR(1) latent state
  */
object StochasticVolatilityKnots {
  /**
    * Sample phi from the autoregressive state space
    * from a conjugate Gaussian distribution
    * @param prior a Gaussian prior distribution
    * @return a function from the current state
    * to the next state with a new value for
    * phi sample from a Gaussian posterior distribution
    */
  def samplePhiConjugate(
    prior:  Gaussian,
    p:      SvParameters,
    alphas: Vector[Double]): Rand[Double] = {

    val pmu = prior.mean
    val psigma = prior.variance

    val sumStates = alphas.
      tail.
      map(at => (at - p.mu)).
      map(x => x * x).
      sum

    val sumStates2 = (alphas.tail.init zip alphas.drop(2)).
      map { case (at1, at) => (at1 - p.mu) * (at - p.mu) }.
      sum

    val prec = 1 / psigma + (1 / p.sigmaEta * p.sigmaEta) * sumStates
    val mean = (pmu / psigma + (1 / p.sigmaEta * p.sigmaEta) * sumStates2) / prec
    val variance = 1 / prec

    Gaussian(mean, sqrt(variance))
  }

  /**
    * Sample the precision of the AR(1) process
    */
  def sampleTau(
    prior:  Gamma,
    p:      SvParameters,
    alphas: Vector[Double]) = {

    val squaredSum = (alphas.tail.init, alphas.drop(2)).zipped.
      map { case (mt, mt1) =>
          val diff = mt1 - (p.mu + p.phi * (mt - p.mu))
          (diff * diff)
      }.sum

    val shape = prior.shape + (alphas.size - 1) * 0.5
    val scale = prior.scale + squaredSum * 0.5

    Gamma(shape, scale)
  }

  /**
    * The log likelihood for the Gaussian approximation 
    * @param state the proposed state for the current block
    * @param observations to observations for the current block
    * @return the log likelihood of the Gaussian approximation 
    */
  def approxLl(
    state: Vector[Double],
    ys:    Vector[(Double, Option[Double])]): Double = {

    val n = ys.length

    val sums = (ys zip state).
      map {
        case ((t, Some(y)), a) => log(y * y) + 1.27 - a
        case _ => 0.0
      }.
      map(x => x * x).
      sum

    -3 * n * 0.5 * log(math.Pi) - (1 / (math.Pi * math.Pi)) * sums
  }

  /**
    * The exact log likelihood of the observations
    * @param state the proposed state for the current block
    * @param ys to observations for the current block
    * @return The exact log likelihood of the observations
    */
  def exactLl(
    state: Vector[Double],
    ys:    Vector[(Double, Option[Double])]): Double = {

    val n = ys.length

    val sums = (ys zip state).
      map {
        case ((t, Some(y)), a) => a + y * y * exp(-a)
        case _ => 0.0
      }.
      sum

    -n * 0.5 * log(2 * math.Pi) - 0.5 * sums
  }

  def transformObs(ys: Vector[(Double, Option[Double])]) = 
    for {
      (t, yo) <- ys
      yt = yo map (y => log(y * y) + 1.27)
    } yield (t, yt)

  type ConditionalFFBS = (FilterAr.SampleState,
    FilterAr.SampleState,
    SvParameters,
    Vector[(Double, Option[Double])]) => Rand[Vector[FilterAr.SampleState]]

  type ConditionalFilter = (FilterAr.SampleState,
    SvParameters,
    Vector[(Double, Option[Double])]) => Rand[Vector[FilterAr.SampleState]]

  type ConditionalSample = (FilterAr.SampleState,
    SvParameters,
    Vector[(Double, Option[Double])]) => Rand[Vector[FilterAr.SampleState]]

  def sampleBlock(
    ys: Vector[(Double, Option[Double])],
    p:  SvParameters,
    filter: ConditionalFFBS) = {

    val transObs = transformObs(ys)

    val prop = (vs: Vector[FilterAr.SampleState]) => {
      filter(vs.head, vs.last, p, transObs)
    }

    val ll = (vs: Vector[FilterAr.SampleState]) => {
      val state = vs.map(_.sample).tail
      exactLl(state, ys) - approxLl(state, ys)
    }

    Metropolis.mAccept[Vector[FilterAr.SampleState]](prop, ll) _
  }

  def sampleEnd(
    ys: Vector[(Double, Option[Double])],
    p:  SvParameters,
    filter: ConditionalFilter) = {

    val transObs = transformObs(ys)

    val prop = (vs: Vector[FilterAr.SampleState]) => {
      filter(vs.head, p, transObs)
    }
    val ll = (vs: Vector[FilterAr.SampleState]) => {
      val state = vs.map(_.sample)
      exactLl(state, ys) - approxLl(state, ys)
    }

    Metropolis.mAccept[Vector[FilterAr.SampleState]](prop, ll) _
  }

  def sampleStart(
    ys: Vector[(Double, Option[Double])],
    p:  SvParameters,
    sampler: ConditionalSample) = {

    val transObs = transformObs(ys)

    val prop = (vs: Vector[FilterAr.SampleState]) => {
      sampler(vs.last, p, transObs)
    }
    val ll = (vs: Vector[FilterAr.SampleState]) => {
      val state = vs.map(_.sample)
      exactLl(state, ys) - approxLl(state, ys)
    }

    Metropolis.mAccept[Vector[FilterAr.SampleState]](prop, ll) _
  }

  // TODO: Fix this
  def sampleStateFold(
    ffbs: ConditionalFFBS,
    filter: ConditionalFilter,
    sampler: ConditionalSample)(
    ys:   Vector[(Double, Option[Double])],
    p:     SvParameters,
    knots: Vector[Int],
    state: Array[FilterAr.SampleState]) = {

    (knots.init zip knots.tail).foldLeft(state.toVector) { case (st, (start, end)) =>
      val selectedObs = ys.slice(start, end)

      if (start == 0) {
        val vs = st.slice(1, end + 1).toVector
        val (res, a) = sampleStart(selectedObs, p, sampler)(vs).draw
        res ++ st.drop(end + 1)
      } else if (end == knots.last) {
        val vs = st.slice(start + 1, end + 1).toVector
        val (res, a) = sampleEnd(selectedObs, p, filter)(vs).draw
        st.take(start + 1) ++ res
      } else {
        val vs = st.slice(start, end + 1).toVector
        val (res, a) = sampleBlock(selectedObs, p, ffbs)(vs).draw
        st.take(start) ++ res ++ st.drop(end)
      }
    }
  }

  def sampleState(
    ffbs: ConditionalFFBS,
    filter: ConditionalFilter,
    sampler: ConditionalSample)(
    ys:   Vector[(Double, Option[Double])],
    p:     SvParameters,
    knots: Vector[Int],
    state: Array[FilterAr.SampleState]): Array[FilterAr.SampleState] = { 

    val finalKnot = knots.last

    for (i <- knots.indices.init) {
      val selectedObs = ys.slice(knots(i), knots(i + 1))

      if (i == 0) {
        val vs = state.slice(1, knots(i + 1) + 1).toVector
        val (res, a) = sampleStart(selectedObs, p, sampler)(vs).draw
        res.copyToArray(state, 0)
      } else if (knots(i + 1) == finalKnot) {
        val vs = state.slice(knots(i) + 1, knots(i + 1) + 1).toVector
        val (res, a) = sampleEnd(selectedObs, p, filter)(vs).draw
        res.copyToArray(state, knots(i) + 1)
      } else {
        val vs = state.slice(knots(i), knots(i + 1) + 1).toVector
        val (res, a) = sampleBlock(selectedObs, p, ffbs)(vs).draw
        res.copyToArray(state, knots(i))
      }
    }

    state
  }

  def discreteUniform(min: Int, max: Int) =
    min + scala.util.Random.nextInt(max - min + 1)

  def sampleStarts(min: Int, max: Int, length: Int) = 
    Stream.continually(discreteUniform(min, max)).
      scanLeft(0)(_ + _).
      takeWhile(_ < length - 1).
      toVector

  /**
    * Sample knot positions by sampling block size 
    * from a uniform distribution between
    * min and max for a sequence of observations of length n
    * @param min the minimum size of a block
    * @param max the maxiumum size of a block
    * @param n the length of the observations
    */
  def sampleKnots(min: Int, max: Int, n: Int): Rand[Vector[Int]] = 
    Rand.always(sampleStarts(min, max, n) :+ n - 1)

  def ffbsAr = FilterAr.conditionalFfbs _

  def filterAr(
    start: FilterAr.SampleState,
    p:     SvParameters,
    transObs: Vector[(Double, Option[Double])]) = {
    val filtered = FilterAr.conditionalFilter(start, p, transObs)
    FilterAr.univariateSample(p, filtered)
  }

  def sampleAr(
    end: FilterAr.SampleState,
    p:     SvParameters,
    transObs: Vector[(Double, Option[Double])]) = {
      val variances = Vector.fill(transObs.size)(Pi * Pi * 0.5)
      val filtered = FilterAr.filterUnivariate(transObs, variances, p)
      FilterAr.conditionalSampler(end, p, filtered)
  }

  def sampleSigmaMetrop(
    prior: InverseGamma,
    p: SvParameters,
    alphas: Vector[Double]) = {

    val proposal = (sigma: Double) =>
      Gaussian(0.0, 0.05).map(z => sigma * exp(z))

    val pos = (newSigma: Double) => {
      val newP = p.copy(sigmaEta = newSigma)
      prior.logPdf(newSigma) +
      StochasticVolatility.arLikelihood(alphas, newP)
    }

    Metropolis.mAccept[Double](proposal, pos) _
  }

  def sampleStepAr(
    priorPhi: Gaussian,
    priorMu:  Gaussian,
    priorSigmaEta: InverseGamma,
    ys:       Vector[(Double, Option[Double])]
  ): StochVolState => Rand[StochVolState] = { st =>
    for {
      knots <- sampleKnots(10, 100, ys.size)
      alphas = sampleState(ffbsAr, filterAr, sampleAr)(ys, st.params,
        knots, st.alphas.toArray).toVector
      state = alphas.map(_.sample)
      newPhi <- samplePhiConjugate(priorPhi, st.params, state)
      newMu <- StochasticVolatility.sampleMu(priorMu, st.params.copy(phi = newPhi), state)
      newSigma <- StochasticVolatility.sampleSigma(priorSigmaEta, st.params.copy(phi = newPhi, mu = newMu), state)
      newP = SvParameters(newPhi, newMu, newSigma)
    } yield StochVolState(newP, alphas)
  }

  def sampleStepArBeta(
    priorPhi:      Beta,
    priorMu:       Gaussian,
    priorSigmaEta: InverseGamma,
    ys:            Vector[(Double, Option[Double])]) = { st: StochVolState =>
    for {
      knots <- sampleKnots(10, 100, ys.size)
      alphas = sampleState(
        ffbsAr, filterAr, sampleAr)(ys, st.params, knots, st.alphas.toArray).toVector
      state = alphas.map(_.sample)
      phi <- StochasticVolatility.samplePhi(priorPhi, st.params, state, 0.05, 100.0)(st.params.phi)
      mu <- StochasticVolatility.sampleMu(priorMu,
        st.params.copy(phi = phi), state)
      se <- StochasticVolatility.sampleSigma(priorSigmaEta,
        st.params.copy(phi = phi, mu = mu), state)
    } yield StochVolState(SvParameters(phi, mu, se), alphas)
  }

  def initialStateAr(
    p:  SvParameters,
    ys: Vector[(Double, Option[Double])]): Rand[Vector[FilterAr.SampleState]] = {

    // transform the observations, centering, squaring and logging
    val transObs = transformObs(ys)
    val vs = Vector.fill(ys.size)(Pi * Pi * 0.5)
    FilterAr.ffbs(p, transObs, vs)
  }

  def sampleParametersAr(
    priorPhi: Gaussian,
    priorMu:  Gaussian,
    priorSigmaEta: InverseGamma,
    ys:       Vector[(Double, Option[Double])]) = {

    // draw from the prior distributions
    val initP = for {
      phi <- priorPhi
      mu <- priorMu
      sigma <- priorSigmaEta
    } yield SvParameters(phi, mu, sigma)
    val params = initP.draw

    val init = StochVolState(params, initialStateAr(params, ys).draw)
    MarkovChain(init)(sampleStepAr(priorPhi, priorMu, priorSigmaEta, ys))
  }

  def sampleArBeta(
    priorPhi:      Beta,
    priorMu:       Gaussian,
    priorSigmaEta: InverseGamma,
    ys:            Vector[(Double, Option[Double])],
    initP:         SvParameters) = {

    val init = StochVolState(initP, initialStateAr(initP, ys).draw)
    MarkovChain(init)(sampleStepArBeta(priorPhi, priorMu, priorSigmaEta, ys))
  }

  def ffbsOu = FilterOu.conditionalFfbs _

  def filterOu(
    start:    FilterAr.SampleState,
    p:        SvParameters,
    transObs: Vector[(Double, Option[Double])]) = {
    val filtered = FilterOu.conditionalFilter(start, p, transObs)
    FilterOu.univariateSample(p, filtered)
  }

  def sampleOu(
    end:      FilterAr.SampleState,
    p:        SvParameters,
    transObs: Vector[(Double, Option[Double])]) = {
    val variances = Vector.fill(transObs.size)(Pi * Pi * 0.5)
    val filtered = FilterOu.filterUnivariate(transObs, variances, p)
    FilterOu.conditionalSampler(end, p, filtered)
  }

  case class OuSvState(
    params:   SvParameters,
    alphas:   Vector[FilterAr.SampleState],
    accepted: DenseVector[Int])

  def sampleStepOu(
    priorPhi:   ContinuousDistr[Double],
    priorMu:    ContinuousDistr[Double],
    priorSigma: ContinuousDistr[Double],
    ys:         Vector[(Double, Option[Double])]): OuSvState => Rand[OuSvState] = { st =>
    for {
      knots <- sampleKnots(10, 100, ys.size)
      alphas = sampleState(
        ffbsOu, filterOu, sampleOu)(ys, st.params, knots, st.alphas.toArray).toVector
      state = alphas.map(x => (x.time, x.sample))
      (newPhi, acceptedPhi) <- StochasticVolatility.samplePhiOu(priorPhi, st.params, state, 0.05)(st.params.phi)
      (newMu, acceptedMu) <- StochasticVolatility.sampleMuOu(priorMu, 0.2,
        st.params.copy(phi = newPhi), state)(st.params.mu)
      (newSigma, acceptedSigma) <- StochasticVolatility.sampleSigmaMetropOu(priorSigma, 0.05,
        st.params.copy(phi = newPhi, mu = newMu), state)(st.params.sigmaEta)
      p = SvParameters(newPhi, newMu, newSigma)
    } yield OuSvState(p, alphas,
      st.accepted + DenseVector(Array(acceptedPhi, acceptedMu, acceptedSigma)))
  }

  def sampleOu(
    priorPhi:   ContinuousDistr[Double],
    priorMu:    ContinuousDistr[Double],
    priorSigma: ContinuousDistr[Double],
    ys:         Vector[(Double, Option[Double])],
    initP:      SvParameters) = {

    val initialState = StochasticVolatility.initialStateOu(initP, ys).draw
    val init = OuSvState(initP, initialState,
      DenseVector.zeros[Int](3))
    MarkovChain(init)(sampleStepOu(priorPhi, priorMu, priorSigma, ys))
  }
}
