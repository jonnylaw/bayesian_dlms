package dlm.examples.urbanobservatory

import dlm.model._
import Dlm._
import dlm.examples._
import cats.implicits._
import cats.Applicative
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.{Gamma, Gaussian, Rand, MarkovChain}
import java.nio.file.Paths
import breeze.numerics.exp
import kantan.csv._
import kantan.csv.ops._

object JointFirstOrder extends App with ObservedData {
  val model = Dlm.outerSumModel(polynomial(1), polynomial(1))
  val initP = Dlm.Parameters(
    DenseMatrix.eye[Double](2),
    DenseMatrix.eye[Double](2),
    DenseVector.zeros[Double](2),
    DenseMatrix.eye[Double](2)
  )

  val iters = GibbsWishart.gibbsSamples(model, Gamma(1.0, 10.0), InverseWishart(1.0, DenseMatrix.eye[Double](2)), initP, data).
    steps.
    take(10000)

  val out = new java.io.File("data/joint_first_order_temperature_humidity_parameters_wishart.csv")
  val headers = rfc.withHeader("V1", "V2", "V3", "V4", "W1", "W2", "W3", "W4")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatParameters(p: Parameters) = {
    (p.v.data ++ p.w.data).toList
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.p))
  }

  writer.close()
}

object JointModel extends App with Models with ObservedData {
  val linear = Dlm.polynomial(1)
  val seasonal = Dlm.seasonal(24, 3)

  val mod = Model(
    (t: Time) => DenseMatrix.horzcat(blockDiagonal(linear.f(t), linear.f(t)), blockDiagonal(seasonal.f(t), seasonal.f(t))),
    (t: Time) => DenseMatrix.horzcat(blockDiagonal(linear.g(t), linear.g(t)), blockDiagonal(seasonal.g(t), seasonal.g(t)))
  )
  val combinedParameters = Dlm.outerSumParameters(initP, initP)

  // couple the linear terms, not the seasonal terms. 
  // Then the "coupled" part will be an Inverse Wishart block, 
  // the other parameters can be updated using a d-inverse gamma step

  // in order to update the coupled "block", we need to isolate just the linear 
  // trend parts of the model:
  // * The 2-dimensions of the 14-dimensional state pertaining to the linear trend
  // * The evolution matrix, G of the linear trend model

  val linearModel = Dlm.outerSumModel(
    Dlm.polynomial(1), 
    Dlm.polynomial(1)
  )
  val seasonalModel = Dlm.outerSumModel(
    Dlm.seasonal(24, 3), 
    Dlm.seasonal(24, 3)
  )

  val iter = (p: Parameters) => for {
    state <- Rand.always(GibbsSampling.sampleState(mod, data, p))
    linearState = state.map { case (t, d) => (t, DenseVector(d(0), d(7))) }
    seasonalState = state.map { case (t, d) => (t, DenseVector.vertcat(d(1 to 6), d(8 to -1))) }
    wCorr <- GibbsWishart.sampleSystemMatrix(InverseWishart(5.0, DenseMatrix.eye[Double](2)), linearModel, linearState)
    wIndep = GibbsSampling.sampleSystemMatrix(Gamma(1.0, 1.0), seasonalModel, seasonalState)
    v = GibbsSampling.sampleObservationMatrix(Gamma(1.0, 1.0), mod, state, data)
    w = blockDiagonal(wCorr, wIndep)
  } yield Parameters(v, w, p.m0, p.c0)

  val iters = MarkovChain(combinedParameters)(iter).steps.take(10000000)

  val out = new java.io.File("data/joint_seasonal_temperature_humidity_parameters.csv")
  val headers = rfc.withHeader(false)
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatParameters(p: Parameters) = {
    (p.v.data ++ p.w.data).toList
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next))
  }

  writer.close()
}
