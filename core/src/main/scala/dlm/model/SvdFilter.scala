package core.dlm.model

import breeze.linalg.{DenseVector, DenseMatrix, diag, svd}
import cats.Traverse
import cats.implicits._

case class SvdState(
  time: Double,
  mt: DenseVector[Double],
  dc: DenseVector[Double],
  uc: DenseMatrix[Double],
  at: DenseVector[Double],
  dr: DenseVector[Double],
  ur: DenseMatrix[Double],
  ft: DenseVector[Double])

/**
  * Perform the Kalman Filter by updating the value of the Singular Value Decomp.
  * of the state covariance matrix, C = UDU^T
  * 
  * https://arxiv.org/pdf/1611.03686.pdf
  */
case class SvdFilter(advState: (SvdState, Double) => SvdState)
    extends FilterTs[SvdState, DlmParameters, DlmModel] {
  import SvdFilter._

  def oneStepForecast(
    f: Double => DenseMatrix[Double],
    at: DenseVector[Double],
    time: Double) = {

    f(time).t * at
  }

  def oneStepMissing(
    fm: DenseMatrix[Double],
    at: DenseVector[Double]) = {

    fm.t * at
  }

  def updateState(
    at: DenseVector[Double],
    dr: DenseVector[Double],
    ur: DenseMatrix[Double],
    p: DlmParameters,
    f: Double => DenseMatrix[Double],
    d: Dlm.Data) = {

    val yt = KalmanFilter.flattenObs(d.observation)

    if (yt.data.isEmpty) {
      (at, dr, ur)
    } else {

      val vm = KalmanFilter.missingV(p.v, d.observation)
      val fm = KalmanFilter.missingF(f, d.time, d.observation)
      val ft = oneStepMissing(fm, at)

      val drInv = dr.map(1.0 / _)
      val root = svd(DenseMatrix.vertcat(vm * fm.t * ur, diag(drInv)))
      val uc = ur * root.rightVectors.t

      val et = yt - ft
      val fv = fm * vm.t * vm
      val dc = root.singularValues.map(1.0 / _)

      val gain = (diag(dc) * uc.t).t * (diag(dc) * uc.t) * fv
      val mt = at + gain * et

      (mt, dc, uc)
    }
  }

  def step(
    mod:      DlmModel,
    p:        DlmParameters)
    (s: SvdState, y: Dlm.Data) = {

    val dt = y.time - s.time
    val st = advState(s, dt)
    val ft = oneStepForecast(mod.f, st.at, y.time) // could add Qt here
    val (mt, dc, uc) = updateState(st.at, st.dr, st.ur, p, mod.f, y)

    SvdState(y.time, mt, dc, uc, st.at, st.dr, st.ur, ft)
  }

  /**
    * Initialise the state of the SVD Filter
    */
  def initialiseState[T[_]: Traverse](
    mod: DlmModel,
    p:   DlmParameters,
    ys:  T[Dlm.Data]) = {

    val root = svd(p.c0)
    val t0 = ys.map(_.time).reduceLeftOption((t0, d) => math.min(t0, d))
    val dc0 = root.singularValues.map(math.sqrt)
    val uc0 = root.rightVectors.t

    val ft = oneStepForecast(mod.f, p.m0, t0.get)

    SvdState(t0.get - 1, p.m0, dc0, uc0, p.m0, dc0, uc0, ft)
  }

  /**
    * Perform the SVD Filter on a traversable collection
    */
  override def filterTraverse[T[_]: Traverse](
    model: DlmModel,
    ys:    T[Dlm.Data],
    p:     DlmParameters): T[SvdState] = {

    val ps = transformParams(p)
    val init = initialiseState(model, ps, ys)
    FilterTs.scanLeft(ys, init, step(model, ps))
  }

  /**
    * Perform the SVD Filter on a traversable collection
    */
  override def filter(
    model: DlmModel,
    ys:    Vector[Dlm.Data],
    p:     DlmParameters): Vector[SvdState] = {

    val ps = transformParams(p)
    val init = initialiseState(model, ps, ys)
    ys.scanLeft(init)(step(model, ps))
  }

  /**
    * Perform the Kalman filter using the decomposition of the DLM parameters
    * using a general traversable collection
    * @param model
    */
  def filterDecompT[T[_]: Traverse](
    model: DlmModel,
    ys:    T[Dlm.Data],
    p:     DlmParameters): T[SvdState] = {

    val init = initialiseState(model, p, ys)
    FilterTs.scanLeft(ys, init, step(model, p))
  }

  def filterDecomp(
    model: DlmModel,
    ys:    Vector[Dlm.Data],
    p:     DlmParameters): Vector[SvdState] = {

    val init = initialiseState(model, p, ys)
    ys.scanLeft(init)(step(model, p))
  }
}

object SvdFilter {
  /**
    * Filter a DLM using the SVD Filter
    */
  def filterDlm[T[_]: Traverse](
    mod: DlmModel,
    ys:  T[Dlm.Data],
    p:   DlmParameters) = {

    SvdFilter(advanceState(p, mod.g)).filterTraverse(mod, ys, p)
  }

  /**
    * Perform the time update by advancing the mean and covariance to the time
    * of the next observation
    * @param g the system evolution matrix
    * @param dt the time difference between observations
    * @param mt the posterior mean at the previous time
    * @param dc the diagonal matrix containing the singular values (eigen values)
    * of the posterior covariance matrix at the previous time
    * @param uc the unitary matrix containing the eigen vectors of the posterior
    * covariance matrix at the previous time step
    * @return
    */
  def advanceState(
    p:   DlmParameters,
    g:   Double => DenseMatrix[Double])
    (s:  SvdState,
     dt: Double) = {

    val (at, dr, ur) = SvdFilter.advState(g, dt, s.mt, s.dc, s.uc, p.w)
    s.copy(at = at, dr = dr, ur = ur)
  }
  
  def advState(
    g:  Double => DenseMatrix[Double],
    dt: Double,
    mt: DenseVector[Double],
    dc: DenseVector[Double],
    uc: DenseMatrix[Double],
    w:  DenseMatrix[Double]) = {
    
    if (dt == 0) {
      (mt, dc, uc)
    } else {
      val at = g(dt) * mt
      val rt = DenseMatrix.vertcat(diag(dc) * uc.t * g(dt).t, w *:* math.sqrt(dt))
      val root = svd(rt)
      val ur = root.rightVectors.t
      val dr = root.singularValues

      (at, dr, ur)
    }
  }

  /**
    * Calculate the square root inverse of a matrix using the Singular value
    * decomposition of a matrix
    * @param m a symmetric positive definite matrix
    * @return the square root inverse of the matrix
    */
  def sqrtInvSvd(m: DenseMatrix[Double]) = {

    val root = svd(m)
    val d = root.singularValues
    val dInv = d.map(e => 1.0 / math.sqrt(e))
    diag(dInv) * root.rightVectors.t
  }

  /**
    * Calculate the square root of a symmetric matrix using eigenvalue
    * decomposition
    * @param m a symmetric positive definite matrix
    * @return the square root of a matrix
    */
  def sqrtSvd(m: DenseMatrix[Double]) = {
    val root = svd(m)
    diag(root.singularValues.map(math.sqrt)) * root.rightVectors.t
  }

  /**
    * Calculate the roots of the observation and noise matrices
    */
  def transformParams(p: DlmParameters): DlmParameters = {
    val sqrtVinv = sqrtInvSvd(p.v)
    val sqrtW = sqrtSvd(p.w)
    p.copy(v = sqrtVinv, w = sqrtW)
  }

  def makeDMatrix(m: Int, n: Int, d: DenseVector[Double]) = {

    DenseMatrix.tabulate(m, n) {
      case (i, j) =>
        if (i == j) d(i) else 0.0
    }
  }
}
