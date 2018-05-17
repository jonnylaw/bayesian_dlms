package core.dlm.model

import breeze.linalg.{DenseVector, DenseMatrix, diag, svd}

/**
  * Perform the Kalman Filter by updating the value of the Singular Value Decomp.
  * of the state covariance matrix, C = UDU^T
  * https://arxiv.org/pdf/1611.03686.pdf
  */
object SvdFilter {
  case class State(
    time: Double,
    mt:   DenseVector[Double],
    dc:   DenseVector[Double],
    uc:   DenseMatrix[Double],
    at:   DenseVector[Double],
    dr:   DenseVector[Double],
    ur:   DenseMatrix[Double],
    ft:   DenseVector[Double]
  )

  def makeDMatrix(
    m: Int,
    n: Int,
    d: DenseVector[Double]) = {

    DenseMatrix.tabulate(m, n){ case (i, j) =>
      if (i == j) d(i) else 0.0 }
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
    g:     Double => DenseMatrix[Double],
    dt:    Double,
    mt:    DenseVector[Double],
    dc:    DenseVector[Double],
    uc:    DenseMatrix[Double],
    p:     Dlm.Parameters
  ) = {
    
    if (dt == 0) {
      (mt, dc, uc)
    } else {
      val at = g(dt) * mt
      val rt = DenseMatrix.vertcat(diag(dc) * uc.t * g(dt).t, p.w *:* math.sqrt(dt))
      val root = svd(rt)
      val ur = root.rightVectors.t
      val dr = root.singularValues

      (at, dr, ur)
    }
  }

  def oneStepForecast(
    f:    Double => DenseMatrix[Double],
    at:   DenseVector[Double],
    time: Double) = {

    f(time).t * at
  }

  def oneStepMissing(
    fm: DenseMatrix[Double],
    at: DenseVector[Double]) = {

    fm.t * at
  }

  def updateState(
    at:       DenseVector[Double],
    dr:       DenseVector[Double],
    ur:       DenseMatrix[Double],
    p:        Dlm.Parameters,
    f:        Double => DenseMatrix[Double],
    d:        Dlm.Data
  ) = {
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
    mod:      Dlm.Model,
    p:        Dlm.Parameters
  )(s: State, y: Dlm.Data) = {

    val dt = y.time - s.time
    val (at, dr, ur) = advanceState(mod.g, dt, s.mt, s.dc, s.uc, p)
    val ft = oneStepForecast(mod.f, at, y.time) // could add Qt here
    val (mt, dc, uc) = updateState(at, dr, ur, p, mod.f, y)

    State(y.time, mt, dc, uc, at, dr, ur, ft)
  }

  /**
    * Initialise the state of the SVD Filter
    */
  def initialiseState(
    mod: Dlm.Model,
    p:   Dlm.Parameters,
    ys:  Vector[Dlm.Data]) = {

    val root = svd(p.c0)
    val t0 = ys.head.time
    val dc0 = root.singularValues.map(math.sqrt)
    val uc0 = root.rightVectors.t
    val (at, dr, ur) = advanceState(mod.g, 0.0, p.m0, dc0, uc0, p)
    val ft = oneStepForecast(mod.f, at, t0)

    State(t0 - 1, p.m0, dc0, uc0, at, dr, ur, ft)
  }

  /**
    * Calculate the square root inverse of a matrix using the Eigenvalue
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

  def filter(
    mod: Dlm.Model,
    ys:  Vector[Dlm.Data],
    p:   Dlm.Parameters) = {

    // calculate roots of covariance matrices and update the parameters
    val sqrtVinv = sqrtInvSvd(p.v)
    val sqrtW = sqrtSvd(p.w)
    val params = p.copy(v = sqrtVinv, w = sqrtW)
    val init = initialiseState(mod, params, ys)

    ys.scanLeft(init)(step(mod, params))
  }
}

