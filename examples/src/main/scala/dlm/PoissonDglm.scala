package examples.dlm

import core.dlm.model._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.MarkovChain
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._

trait PoissonDglm {
  val mod = Dglm.poisson(Dlm.polynomial(1))
  val params = DlmParameters(DenseMatrix(2.0),
                             DenseMatrix(0.01),
                             DenseVector(0.0),
                             DenseMatrix(1.0))
}

trait PoissonData {
  val rawData = Paths.get("core/data/poisson_dglm.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Dlm.Data(a.head, DenseVector(a(1).some))
  }.toVector
}

object SimulatePoissonDglm extends App with PoissonDglm {
  val sims = Dglm.simulateRegular(mod, params, 1.0).steps.take(1000)

  val out = new java.io.File("core/data/poisson_dglm.csv")
  val header = rfc.withHeader("time", "observation", "state")
  val writer = out.asCsvWriter[List[Double]](header)

  def formatData(d: (Dlm.Data, DenseVector[Double])) = d match {
    case (Dlm.Data(t, y), x) =>
      t :: KalmanFilter.flattenObs(y).data.toList ::: x.data.toList
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

object PoissonDglmGibbs extends App with PoissonDglm with PoissonData {
  val n = 200
  val model = DlmModel(mod.f, mod.g)
  val priorW = InverseGamma(11.0, 1.0)

  val mcmcStep = (s: List[(Double, DenseVector[Double])], p: DlmParameters) =>
    for {
      w <- GibbsSampling.sampleSystemMatrix(priorW, s.toVector, model.g)
      (ll, state) <- ParticleGibbs.sample(n, params, model, data.toList)
    } yield (state, DlmParameters(p.v, w, p.m0, p.c0))

  val initState = ParticleGibbs.sample(n, params, model, data.toList).draw._2
  val iters = MarkovChain((initState, params)) { case (x, p) => mcmcStep(x, p) }.steps
    .map(_._2)
    .take(10000)

  val out = new java.io.File("core/data/poisson_dglm_gibbs.csv")
  val writer = out.asCsvWriter[Double](rfc.withHeader("W"))

  def formatParameters(p: DlmParameters) = {
    (p.w.data(0))
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next))
  }

  writer.close()
}
