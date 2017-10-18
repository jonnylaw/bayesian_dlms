package dlm.examples.urbanobservatory

import dlm.model._
import Dlm._
import dlm.examples._
import MetropolisHastings._
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

  val iters = GibbsWishart.gibbsSamples(model, InverseGamma(1.0, 10.0), InverseWishart(1.0, DenseMatrix.eye[Double](2)), initP, data).
    steps.
    take(1000000)

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

trait JointModel extends Models {
  val linear = Dlm.polynomial(1)
  val seasonal = Dlm.seasonal(24, 3)

  val mod = Model(
    (t: Time) => DenseMatrix.vertcat(
      blockDiagonal(linear.f(t), linear.f(t)), 
      blockDiagonal(seasonal.f(t), seasonal.f(t))
    ),
    (t: Time) => blockDiagonal(
      blockDiagonal(linear.g(t), linear.g(t)), 
      blockDiagonal(seasonal.g(t), seasonal.g(t))
    )
  )
  val combinedParameters = Dlm.outerSumParameters(initP, initP)
}

object SimJointModel extends App with JointModel {
  val sims = simulate(0, mod, combinedParameters).
    steps.
    take(1000)

  val out = new java.io.File("data/simulated_humid_temp.csv")
  val header = rfc.withHeader("time", "observation1", "observation2", "state1", "state2", 
    "state3", "state4", "state5", "state6", "state7", "state8",
    "state9", "state10", "state11", "state12", "state13", "state14")
  val writer = out.asCsvWriter[List[Double]](header)

  def formatData(d: (Data, DenseVector[Double])) = d match {
    case (Data(t, y), x) =>
      List(t.toDouble) ++ y.map(_.data).get ++ x.data
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

object FitJointModel extends App with JointModel with ObservedData {
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

  /**
    * Normalise the observed Data
    */
  def normalise(xs: Array[Option[Double]]) = {
    val mv = breeze.stats.meanAndVariance(xs.flatten)
    for {
      x <- xs
      mean = mv.mean
      sd = breeze.numerics.sqrt(mv.variance)
    } yield x.map(a => (a - mean) / sd)
  }

  val humidity = normalise(data.map { case Data(_, y) => y.map(a => a(0)) })
  val temperature = normalise(data.map { case Data(_, y) => y.map(a => a(1)) })

  val normalisedData = (humidity zip temperature).zipWithIndex.
    map { case ((humid, temp), time) => Data(time, for {
      h <- humid
      t <- temp
    } yield DenseVector(h, t)) }

  val iter = (p: Parameters) => for {
    state <- Rand.always(GibbsSampling.sampleState(mod, normalisedData, p))

    // seperate the states
    linearState = state.map { case (t, d) => (t, DenseVector(d(0), d(1))) }
    seasonalState = state.map { case (t, d) => (t, DenseVector.vertcat(d(2 to -1))) }

    // update W in two stages
    wCorr <- GibbsWishart.sampleSystemMatrix(InverseWishart(5.0, DenseMatrix.eye[Double](2)), linearModel, linearState)
    wIndep <- GibbsSampling.sampleSystemMatrix(InverseGamma(1.0, 1.0), seasonalModel, seasonalState)
    w = blockDiagonal(wCorr, wIndep)

    // update the observation variance
    v <- GibbsSampling.sampleObservationMatrix(InverseGamma(1.0, 1.0), mod, state, data)

  } yield Parameters(v, w, p.m0, p.c0)

  val iters = MarkovChain(combinedParameters)(iter).steps.
    take(10000)

  val out = new java.io.File("data/joint_seasonal_temperature_humidity_parameters.csv")
  val headers = rfc.withHeader(false)
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatParameters(p: Parameters) = {
    (DenseVector.vertcat(diag(p.v), diag(p.w))).data.toList ++ List(p.w(0,1), p.w(1, 0))
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next))
  }

  writer.close()
}
