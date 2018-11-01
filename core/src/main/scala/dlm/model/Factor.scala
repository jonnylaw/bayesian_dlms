// package dlm.core.model

// import breeze.stats.distributions._
// import breeze.linalg.{DenseVector, DenseMatrix, diag}
// import cats.Applicative
// import cats.implicits._

// /**
//   * A DLM with a factor structure for the system matrix
//   */
// object FactorDlm {
//   case class Parameters(
//     beta: DenseMatrix[Double],
//     factors: Vector[DenseVector[Double]],
//     sigmax: DenseMatrix[Double],
//     v: DenseMatrix[Double]
//   )

//   case class State(
//     theta: Vector[SamplingState],
//     p: Parameters
//   )

//   def calculateVariance(
//     fs:   Vector[DenseVector[Double]],
//     beta: DenseMatrix[Double],
//     v:    DenseMatrix[Double]): Vector[DenseMatrix[Double]] = 
//     fs map (f => beta * f * beta.t + v)

//   def dlmParameters(p: Parameters) =
//     DlmParameters(
//       v = p.v,
//       w = DenseMatrix.eye[Double](p.beta.rows),
//       DenseVector.zeros[Double](p.beta.rows),
//       DenseMatrix.eye[Double](p.beta.rows) * 10.0)

//   def sampleFactors(
//     ys: Vector[Data],
//     p: Parameters): Rand[Vector[DenseVector[Double]]] = {

//     val precY = p.sigmax.map(1.0 / _)
//     val obs = FactorSv.encodePartiallyMissing(observations)
//     val identity = DenseMatrix.eye[Double](p.beta.rows)
//     val beta = p.beta

//     // sample factors independently
//     val res = for {
//       ((time, ys), at) <- obs zip volatility
//       prec = identity + (beta.t * precY * beta)
//       mean = ys.map(y => prec \ (beta.t * (precY * y)))
//       sample = mean map (m => rnorm(m, prec).draw)
//     } yield (time, sample)

//     Rand.always(res)
//   }

//   def sampleStep(
//     priorBeta: Gaussian,
//     priorFactor: Gaussian,
//     priorSigma: InverseGamma,
//     priorV: InverseGamma,
//     model: Dlm,
//     ys: Vector[Data])(s: State): Rand[State] = {

//     // extract dimensions of observations and factors
//     val p = s.p.beta.rows
//     val k = s.p.beta.cols

//     for {
//       fs1 <- sampleFactors(ys, s.p)
//       ws = calculateVariance(fs1, s.p.beta, s.p.sigmax)
//       dlmP = dlmParameters(s.p)
//       theta <- DlmFsvSystem.ffbs(model, ys, dlmP, ws)
//       beta <- FactorSv.sampleBeta(priorBeta, ys, p, k, fs1, s.p.sigmax)
//       state = theta.map(s => (s.time, s.sample))
//       newV <- GibbsSampling.sampleObservationMatrix(priorV, model.f,
//                                                     ys.map(_.observation), state)
//       sigmax = FactorSv.sampleSigma(priorSigma, theta, model.g, s.p)
//       newP = Parameters(beta, fs1, sigmax, ws)
//     } yield State(theta.toVector, newP)
//   }

//   /**
//     * Initialise the state of the DLM FSV system Model
//     * by initialising variance matrices for the system, performing FFBS for
//     * the mean state
//     * @param params parameters of the DLM FSV system model
//     * @param ys time series of observations
//     * @param dlm the dlm specification
//     */
//   def initialise(
//     params: DlmFsvParameters,
//     ys:     Vector[Data],
//     dlm:    Dlm) = {

//     val k = params.fsv.beta.cols
//     val parameters = params.dlm

//     // initialise the variances of the system
//     val theta = SvdSampler.ffbs(dlm, ys, dlmParameters(parameters)).draw
//     val thetaObs = theta.map { ss => Data(ss.time, ss.sample.map(_.some)) }
//     val fs = FactorSv.initialiseStateAr(params.fsv, thetaObs, k)

//     State(params, theta.toVector, fs.factors, fs.volatility)
//   }

//   def samplePrior(
//     priorBeta: Gaussian,
//     priorFactor: Gaussian,
//     priorSigma: InverseGamma,
//     priorV: InverseGamma,
//     p: Int,
//     k: Int): Rand[Parameters] = {

//     val initP = for {
//       s <- Applicative[Rand].replicateA(p, priorSigma)
//       sigmax = diag(DenseVector(s.toArray))
//       v <- Applicative[Rand].replicateA(p, priorV)
//       vs = diag(DenseVector(v.toArray))
//       beta = FactorSv.buildBeta(p, k, priorBeta.draw)
//       f <- Applicative[Rand].replicateA(k, priorFactor)
//       fs = DenseVector(f.toArray)
//     } yield Parameters(beta, fs, sigmax, vs)

//   }

//   def sample(
//     priorBeta: Gaussian,
//     priorFactor: Gaussian,
//     priorSigma: InverseGamma,
//     priorV: InverseGamma,
//     model: Dlm,
//     ys: Vector[Data],
//     k: Int): Process[State] = {

//     // initialise the latent state
//     val p = ys.head.observation.size

//     val initP = samplePrior(priorBeta, priorFactor, priorV, p, k)
//     val init = initialise(initP, ys, dlm)

//     MarkovChain(init)(sampleStep(priorBeta, priorFactor, priorSigma, priorV, ys, dlm))

//   }
// }
