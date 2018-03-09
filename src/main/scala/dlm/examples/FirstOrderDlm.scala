package dlm.examples

import dlm.model._
import Dlm._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{Gaussian, MarkovChain, Rand}
import java.nio.file.Paths
import java.io.File
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._

trait FirstOrderDlm {
  val mod = Dlm.Model(
    f = (t: Double) => DenseMatrix((1.0)), 
    g = (t: Double) => DenseMatrix((1.0))
  )
  val p = Dlm.Parameters(
    DenseMatrix(3.0), 
    DenseMatrix(1.0), 
    DenseVector(0.0), 
    DenseMatrix(1.0))
}

trait SimulatedData {
  val rawData = Paths.get("data/first_order_dlm.csv")
  val reader = rawData.asCsvReader[(Double, Double, Double)](rfc.withHeader)
  val data = reader.
    collect { 
      case Success(a) => Data(a._1, DenseVector(a._2.some))
    }.
    toVector
}

object SimulateDlm extends App with FirstOrderDlm {
  val sims = simulateRegular(0, mod, p,1.0).
    steps.
    take(100)

  val out = new java.io.File("data/first_order_dlm.csv")
  val headers = rfc.withHeader("time", "observation", "state")
  val writer = out.asCsvWriter[List[Double]](headers)

  def formatData(d: (Data, DenseVector[Double])) = d match {
    case (Data(t, y), x) =>
      t :: KalmanFilter.flattenObs(y).data.toList ::: x.data.toList
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

object FilterDlm extends App with FirstOrderDlm with SimulatedData {
  val filtered = KalmanFilter.filter(mod, data, p)

  val out = new java.io.File("data/first_order_dlm_filtered.csv")

  def formatFiltered(f: KalmanFilter.State) = {
    (f.time, f.mt(0), f.ct.data(0), f.ft.map(_(0)), f.qt.map(_.data(0)))
  }
  val headers = rfc.withHeader("time", "state_mean", "state_variance", "one_step_forecast", "one_step_variance")

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object SmoothDlm extends App with FirstOrderDlm with SimulatedData {
  val filtered = KalmanFilter.filter(mod, data, p)
  val smoothed = Smoothing.backwardsSmoother(mod)(filtered)

  val out = new java.io.File("data/first_order_dlm_smoothed.csv")

  def formatSmoothed(s: Smoothing.SmoothingState) = 
    (s.time, s.mean(0), s.covariance.data(0))

  out.writeCsv(smoothed.map(formatSmoothed),
    rfc.withHeader("time", "smoothed_mean", "smoothed_variance"))
}
 
object GibbsParameters extends App with FirstOrderDlm with SimulatedData {
  val iters = GibbsSampling.sample(mod, InverseGamma(4.0, 9.0), InverseGamma(6.0, 5.0), p, data).
    steps.
    take(10000)

  val out = new java.io.File("data/first_order_dlm_gibbs.csv")
  val writer = out.asCsvWriter[(Double, Double, Double, Double)](rfc.withHeader("V", "W", "m0", "c0"))

  def formatParameters(p: Parameters) = {
    (p.v.data(0), p.w.data(0), p.m0.data(0), p.c0.data(0))
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next.p))
  }

  writer.close()
}

/**
  * Run Particle Gibbs Sampling on the first order DLM
  */
object ParticleGibbsFo extends App with FirstOrderDlm with SimulatedData {
  // choose number of particles and sample an initial state
  val n = 1000
  val initFilter = ParticleFilter.filter(mod, data, p, n)
  val conditionedState = ParticleGibbs.sampleState(
    initFilter.map(d => d.state.map((d.time, _)).toList).toList, 
    initFilter.last.weights.toList
  ).draw

  val filter = ParticleGibbs.filter(1000, p, mod, data.toList) _
  val gibbsFilter = MarkovChain(conditionedState)(x => filter(x).map(_._2)).
    steps.
    take(100)

  def writeFiltering(file: String, state: Iterator[List[Double]]) = {
    val out = new java.io.File(file)
    val writer = out.asCsvWriter[List[Double]](rfc.withHeader(false))

    while (state.hasNext) {
      writer.write(state.next)
    }

    writer.close()
  }

  def formatState(s: List[(Double, DenseVector[Double])]): List[Double] = {
    s.map(x => x._2.data.head)
  }

  writeFiltering("data/particle_gibbs.csv", gibbsFilter.map(formatState))
}

object ParticleGibbsAncestorFo extends App with FirstOrderDlm with SimulatedData {
  // choose number of particles and sample an initial state
  val n = 1000
  val initFilter = ParticleFilter.filter(mod, data, p, n)
  val conditionedState = ParticleGibbs.sampleState(
    initFilter.map(d => d.state.map((d.time, _)).toList).toList, 
    initFilter.last.weights.toList
  ).draw

  val filter = ParticleGibbsAncestor.filter(n, p, mod, data.toList) _
  val ancestorFilter = MarkovChain(conditionedState)(x => filter(x).map(_._2)).
    steps.
    take(100)

  def writeFiltering(file: String, state: Iterator[List[Double]]) = {
    val out = new java.io.File(file)
    val writer = out.asCsvWriter[List[Double]](rfc.withHeader(false))

    while (state.hasNext) {
      writer.write(state.next)
    }

    writer.close()
  }

  def formatState(s: List[(Double, DenseVector[Double])]): List[Double] = {
    s.map(x => x._2.data.head)
  }

  writeFiltering("data/particle_gibbs_ancestor.csv", 
    ancestorFilter.map(formatState))
}
