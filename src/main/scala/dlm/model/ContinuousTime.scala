package dlm.model

import breeze.linalg._

object ContinuousTime {
  type TimeCont = Double
  type TimeIncrement = Double

  def matrixExponential(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    val eig = eigSym(m)

    eig.eigenvectors * diag(eig.eigenvalues.mapValues(math.exp)) * inv(eig.eigenvectors)
  }

  // The transition probability at time t is p(t) = p(x_t | G x_{t-1}, W) =
  // N(G x_{t-1, W) for a single timestep dt = 1
  // how about for a timestep of dt < 1
  // N(x_{t-1}, dt W) for G = I, a random walk
  def randomWalkStep(
    x: DenseVector[Double], 
    w: DenseMatrix[Double])(dt: TimeIncrement) = {

    MultivariateGaussianSvd(x, dt * w)
  }

  // What about a rotation matrix G = R
  // Infinitessimal rotation

  // w small, then cos(w) ~ 1, sin(w) ~ w and R = ((1, -w), (w, 1))
  // Then for a small rotation we have (I + Adw) x
  // I think we need the time Increment of the step regardless
  // w = 2pi (t + dt) % t / T
  // How do we calculate two steps, or half a step of a rotation?
  // matrix eponential?
  def angle(period: Int)(t: TimeCont, dt: TimeIncrement): Double = {
    2 * math.Pi * ((t + dt) % period) / period
  }

  def seasonalStep(
    period:    Int,
    harmonics: Int,
    x:         DenseVector[Double],
    w:         DenseMatrix[Double])
    (dt:       TimeIncrement,
    t:         TimeCont) = {

    val matrices = (1 to harmonics).
      map(h => Dlm.rotationMatrix(h * angle(period)(t, dt)))
    val g = matrices.reduce(Dlm.blockDiagonal)

    MultivariateGaussianSvd(g * x, dt * w)
  }
}
 
