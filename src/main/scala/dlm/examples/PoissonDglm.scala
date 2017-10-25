package dlm.examples

import dlm.model._
import ParticleGibbs._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{Poisson, MarkovChain}
import java.nio.file.Paths
import math.exp
import java.io.File
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._

trait PoissonDglm {
  val mod = Dglm.Model(
    observation = (x, v) => Poisson(exp(x(0))).map(DenseVector(_)),
    f = (t: Time) => DenseMatrix((1.0)), 
    g = (t: Time) => DenseMatrix((1.0))
  )
  val params = Dlm.Parameters(
    DenseMatrix(2.0), 
    DenseMatrix(0.01), 
    DenseVector(0.0), 
    DenseMatrix(1.0))
}

trait PoissonData {
  val rawData = Paths.get("data/poisson_dglm.csv")
  val reader = rawData.asCsvReader[(Time, Double, Double)](rfc.withHeader)
  val data = reader.
    collect { 
      case Success(a) => Dlm.Data(a._1, Some(a._2).map(DenseVector(_)))
    }.
    toArray
}

object SimulatePoissonDglm extends App with PoissonDglm {
  val sims = Dglm.simulate(mod, params).
    steps.
    take(1000)

  val out = new java.io.File("data/poisson_dglm.csv")
  val writer = out.asCsvWriter[(Time, Option[Double], Double)](rfc.withHeader("time", "observation", "state"))

  def formatData(d: (Dlm.Data, DenseVector[Double])) = d match {
    case (Dlm.Data(t, y), x) =>
      (t, y.map(x => x(0)), x(0))
  }

  while (sims.hasNext) {
    writer.write(formatData(sims.next))
  }

  writer.close()
}

/**
  * Use Particle Gibbs to determine the parameters of the poisson DGLM
  */
object PoissonDglmGibbs extends App with PoissonDglm with PoissonData {
  val n = 200
  val model = Dlm.Model(mod.f, mod.g)
  val initFilter = ParticleFilter.filter(model, data, params, n, Dglm.poisson)
  val conditionedState = ParticleGibbs.ancestorResampling(
    initFilter.map(d => d.state.map((d.time, _)).toList).toList, 
    initFilter.last.weights.toList
  ).draw

  val filter = ParticleGibbs.filter(1000, params, Dglm.poisson, model, data.toList) _
  val priorW = InverseGamma(5.0, 4.0)

  val mcmcStep = (s: LatentState, p: Dlm.Parameters) => for {
    state <- ParticleGibbs.pgas(filter(s))
    w <- GibbsSampling.sampleSystemMatrix(priorW, model, state.toArray)
  } yield (state, Dlm.Parameters(p.v, w, p.m0, p.c0))

  val iters = MarkovChain((conditionedState, params)){ case (x, p) => mcmcStep(x, p) }.
    steps.
    map(_._2).
    take(10000)

  val out = new java.io.File("data/poisson_dglm_gibbs.csv")
  val writer = out.asCsvWriter[Double](rfc.withHeader("W"))

  def formatParameters(p: Dlm.Parameters) = {
    (p.w.data(0))
  }

  // write iters to file
  while (iters.hasNext) {
    writer.write(formatParameters(iters.next))
  }

  writer.close()
}
