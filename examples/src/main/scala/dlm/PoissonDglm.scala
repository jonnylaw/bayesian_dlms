package examples.dlm

import dlm.core.model._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{MarkovChain, MultivariateGaussian}
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._

trait PoissonDglm {
  val mod = Dglm.poisson(Dlm.polynomial(1))
  val params = DlmParameters(
    DenseMatrix(2.0),
    DenseMatrix(0.05),
    DenseVector(0.0),
    DenseMatrix(1.0))
}

trait PoissonData {
  val rawData = Paths.get("examples/data/poisson_dglm.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Data(a.head, DenseVector(a(1).some))
  }.toVector
}

object SimulateState extends App with PoissonDglm {
  val init = MultivariateGaussian(params.m0, params.c0).draw
  val sims = MarkovChain((0.0, init)){ case (t, x) =>
    for {
      x1 <- Dlm.stepState(Dlm.polynomial(1), params, x, 1.0)
    } yield (t + 1.0, x1) }.
    steps.
    take(1000)

  val out = new java.io.File("examples/data/latent_state.csv")
  val header = rfc.withHeader("time", "observation", "state")
  val writer = out.asCsvWriter[List[Double]](header)

  def formatData(d: (Double, DenseVector[Double])) = d match {
    case (t, x) => t :: x.data.toList
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

object SimulatePoissonDglm extends App with PoissonDglm {
  val rawData = Paths.get("examples/data/latent_state.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val state = reader.collect {
    case Right(a) => (a.head, DenseVector(a(1)))
  }.toVector

  val sims = state.map { case (t, x) => (t, mod.observation(x, params.v).draw, x) }

  val out = new java.io.File("examples/data/poisson_dglm.csv")
  val header = rfc.withHeader("time", "observation", "state")

  def formatData(d: (Double, DenseVector[Double], DenseVector[Double])) = d match {
    case (t, y, x) =>
      t :: y.data.toList ::: x.data.toList
  }

  out.writeCsv(sims.map(formatData), header)
}

object SimulateZeroInflated extends App {
  val mod = Dglm.zip(Dlm.polynomial(1))
  val params = DlmParameters(DenseMatrix(Dglm.logit(0.2)),
                             DenseMatrix(0.5),
                             DenseVector(0.0),
                             DenseMatrix(1.0))

  val rawData = Paths.get("examples/data/latent_state.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val state = reader.collect {
    case Right(a) => (a.head, DenseVector(a(1)))
  }.toVector

  val sims = state.map { case (t, x) => (t, mod.observation(x, params.v).draw, x) }

  val out = new java.io.File("examples/data/zip_dglm.csv")
  val header = rfc.withHeader("time", "observation", "state")

  def formatData(d: (Double, DenseVector[Double], DenseVector[Double])) = d match {
    case (t, y, x) =>
      t :: y.data.toList ::: x.data.toList
  }

  out.writeCsv(sims.map(formatData), header)
}

object SimulateNegativeBinomial extends App {
  val mod = Dglm.negativeBinomial(Dlm.polynomial(1))
  val params = DlmParameters(DenseMatrix(math.log(1.0)),
                             DenseMatrix(0.5),
                             DenseVector(0.0),
                             DenseMatrix(1.0))

  val rawData = Paths.get("examples/data/latent_state.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val state = reader.collect {
    case Right(a) => (a.head, DenseVector(a(1)))
  }.toVector

  val sims = state.map { case (t, x) => (t, mod.observation(x, params.v).draw, x) }

  val out = new java.io.File("examples/data/negative_binomial_dglm.csv")
  val header = rfc.withHeader("time", "observation", "state")

  def formatData(d: (Double, DenseVector[Double], DenseVector[Double])) = d match {
    case (t, y, x) =>
      t :: y.data.toList ::: x.data.toList
  }

  out.writeCsv(sims.map(formatData), header)
}

object FilterPoisson extends App with PoissonDglm with PoissonData {
  val advState = (s: PfState, dt: Double) => s
  val filtered = ParticleFilter(500, ParticleFilter.metropolisResampling(10)).
    filter(mod, data, params)

  val out = new java.io.File("examples/data/poisson_filtered_metropolis.csv")
  val header = rfc.withHeader("time", "state_mean", "state_var")

  def formatData(s: PfState) = {
    List(s.time) ++ LiuAndWestFilter.meanState(s.state).data.toList ++
    LiuAndWestFilter.varState(s.state).data.toList
  }

  out.writeCsv(filtered.map(formatData), header)
}

// object PoissonParameters extends App
//     with PoissonDglm with PoissonData {


// }

// object PoissonDglmGibbs extends App with PoissonDglm with PoissonData {
//   val n = 200
//   val model = Dlm(mod.f, mod.g)
//   val priorW = InverseGamma(11.0, 1.0)

//   val mcmcStep = (s: List[SamplingState], p: DlmParameters) =>
//     for {
//       w <- GibbsSampling.sampleSystemMatrix(priorW, s.toVector, model.g)
//       (ll, state) <- ParticleGibbs.sample(n, params, model, data.toList)
//     } yield (state, DlmParameters(p.v, w, p.m0, p.c0))

//   val initState = ParticleGibbs.sample(n, params, model, data.toList).draw._2
//   val iters = MarkovChain((initState, params)) { case (x, p) => mcmcStep(x, p) }.steps
//     .map(_._2)
//     .take(10000)

//   val out = new java.io.File("examples/data/poisson_dglm_gibbs.csv")
//   val writer = out.asCsvWriter[Double](rfc.withHeader("W"))

//   def formatParameters(p: DlmParameters) = {
//     (p.w.data(0))
//   }

//   // write iters to file
//   while (iters.hasNext) {
//     writer.write(formatParameters(iters.next))
//   }

//   writer.close()
// }
