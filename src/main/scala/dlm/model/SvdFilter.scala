package dlm.model

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
    sqrtW: DenseMatrix[Double]
  ) = {
    
    if (dt == 0) {
      (mt, dc, uc)
    } else {
      val at = g(dt) * mt
      val rt = DenseMatrix.vertcat(diag(dc) * (g(dt) * uc).t, sqrtW *:* math.sqrt(dt))
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
    sqrtVInv: DenseMatrix[Double],
    f:        Double => DenseMatrix[Double],
    d:        Dlm.Data
  ) = {
    val yt = KalmanFilter.flattenObs(d.observation)

    if (yt.data.isEmpty) {
      (at, dr, ur)
    } else {

      val vm = KalmanFilter.missingV(sqrtVInv, d.observation)
      val fm = KalmanFilter.missingF(f, d.time, d.observation)
      val ft = oneStepMissing(fm, at)

      val drInv = dr.map(1.0 / _)
      val gain = svd(DenseMatrix.vertcat(vm * fm.t * ur, diag(drInv)))
      val uc = ur * gain.rightVectors.t

      val dc = gain.singularValues.map(1.0 / _)
      val et = yt - ft
      val fv = fm * vm.t * vm
      val mt = at + (diag(dc) * uc.t).t * (diag(dc) * uc.t) * fv * et

      (mt, dc, uc)
    }
  }

  def step(
    mod:      Dlm.Model,
    p:        Dlm.Parameters,
    sqrtVInv: DenseMatrix[Double],
    sqrtW:    DenseMatrix[Double]
  )(s: State, y: Dlm.Data) = {

    val dt = y.time - s.time
    val (at, dr, ur) = advanceState(mod.g, dt, s.mt, s.dc, s.uc, sqrtW)
    val ft = oneStepForecast(mod.f, at, y.time) // could add Qt here
    val (mt, dc, uc) = updateState(at, dr, ur, sqrtVInv, mod.f, y)

    State(y.time, mt, dc, uc, at, dr, ur, ft)
  }

  /**
    * Initialise the state of the SVD Filter
    */
  def initialiseState(
    mod: Dlm.Model,
    p:   Dlm.Parameters,
    ys:  Vector[Dlm.Data],
    sqrtW: DenseMatrix[Double]) = {

    val root = svd(p.c0)
    val t0 = ys.head.time
    val (at, dr, ur) = advanceState(mod.g, 0.0, p.m0,
      root.singularValues.map(math.sqrt), root.rightVectors.t, sqrtW)
    val ft = oneStepForecast(mod.f, at, t0)

    State(t0 - 1, p.m0, root.singularValues.map(math.sqrt), root.rightVectors.t,
      at, dr, ur, ft)
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

    val sqrtVinv = sqrtInvSvd(p.v)
    val sqrtW = sqrtSvd(p.w)
    val init = initialiseState(mod, p, ys, sqrtW)

    ys.scanLeft(init)(step(mod, p, sqrtVinv, sqrtW))
  }
}

