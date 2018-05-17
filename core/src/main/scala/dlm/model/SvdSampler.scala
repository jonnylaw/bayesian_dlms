package core.dlm.model

import breeze.linalg.{DenseVector, DenseMatrix, diag, svd}
import breeze.stats.distributions.{Gaussian, Rand}

/**
  * Backward Sampler utilising the SVD for stability
  * TODO: Check this
  */
object SvdSampler {
  case class State(
    time:  Double,
    theta: DenseVector[Double],
    at1:   DenseVector[Double])

  /**
    * Perform a single step in the backward sampler using the SVD
    */
  def step(
    mod:      Dlm.Model,
    sqrtWInv: DenseMatrix[Double]
  )(st:       SvdFilter.State,
    ss:       State) = {

    val dt = ss.time - st.time
    val dcInv = st.dc.map(1.0 / _)
    val root = svd(DenseMatrix.vertcat(sqrtWInv * mod.g(dt) * st.uc, diag(dcInv)))
    val uh = st.uc * root.rightVectors.t
    val dh = root.singularValues.map(1.0 / _)

    val gWinv = mod.g(dt).t * sqrtWInv.t * sqrtWInv
    val du = diag(dh) * uh.t
    val h = st.mt + du.t * du * gWinv * (ss.theta - ss.at1)

    State(st.time, rnorm(h, dh, uh).draw, st.at)
  }

  def initialise(
    filtered: Array[SvdFilter.State]) = {

    val lastState = filtered.last
    State(lastState.time,
      SvdSampler.rnorm(lastState.mt, lastState.dc, lastState.uc).draw,
      lastState.at)
  }

  /**
    * Given a vector containing the SVD filtered results, perform backward sampling
    * @param mod a DLM specification
    * @param w the system error matrix
    * @param st 
    * @return
    */
  def sample(
    mod: Dlm.Model,
    w:   DenseMatrix[Double],
    st:  Vector[SvdFilter.State]): Vector[(Double, DenseVector[Double])] = {

    val sqrtWinv = SvdFilter.sqrtInvSvd(w)
    val init = initialise(st.toArray)

    st.init.scanRight(init)(step(mod, sqrtWinv)).
      map(a => (a.time, a.theta))
  }

  /**
    * Perform forward filtering backward sampling
    */
  def ffbs(
    mod: Dlm.Model,
    ys:  Vector[Dlm.Data],
    p:   Dlm.Parameters) = {
   
    val filtered = SvdFilter.filter(mod, ys, p)
    Rand.always(sample(mod, p.w, filtered))
  }

  /**
    * Simulate from a normal distribution given the right vectors and
    * singular values of the covariance matrix
    * @param mu the mean of the multivariate normal distribution
    * @param d the square root of the diagonal in the SVD of the 
    * Error covariance matrix C_t
    * @param u the right vectors of the SVDfilter
    * @return a DenseVector sampled from the Multivariate Normal distribution with
    * mean mu and covariance u d^2 u^T
    */
  def rnorm(
    mu: DenseVector[Double],
    d:  DenseVector[Double],
    u:  DenseMatrix[Double]) = new Rand[DenseVector[Double]] {

    def draw = {
      val z = DenseVector.rand(mu.size, Gaussian(0, 1))
      mu + (u * diag(d) * z)
    }
  }

  def meanState(sampled: Seq[Seq[(Double, DenseVector[Double])]]) = {
    sampled.
      transpose.
      map(s => (s.head._1, s.map(_._2).reduce(_ + _) /:/ sampled.size.toDouble)).
      map { case (t, s) => List(t, s(0)) }
  }

  def intervalState(
    sampled: Seq[Seq[(Double, DenseVector[Double])]],
    interval: Double = 0.95): Seq[(Double, (DenseVector[Double], DenseVector[Double]))] = ???
}