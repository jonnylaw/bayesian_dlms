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
  val mod = Dglm.poisson(Dlm.polynomial(1))
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
      case Success(a) => Data(a._1, Some(a._2).map(DenseVector(_)))
    }.
    toArray
}

object SimulatePoissonDglm extends App with PoissonDglm {
  val sims = Dglm.simulate(mod, params).
    steps.
    take(1000)

  val out = new java.io.File("data/poisson_dglm.csv")
  val header = rfc.withHeader("time", "observation", "state")
  val writer = out.asCsvWriter[(Time, Option[Double], Double)](header)

  def formatData(d: (Data, DenseVector[Double])) = d match {
    case (Data(t, y), x) =>
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
  val initFilter = ParticleFilter.filter(model, data, params, n)
  val conditionedState = ParticleGibbs.sampleState(
    initFilter.map(d => d.state.map((d.time, _)).toList).toList, 
    initFilter.last.weights.toList
  ).draw

  val priorW = InverseGamma(11.0, 1.0)

  val mcmcStep = (s: LatentState, p: Dlm.Parameters) => for {
    w <- GibbsSampling.sampleSystemMatrix(priorW, model.g, s.toArray)
    (ll, state) <- ParticleGibbs.filter(n, params, model, data.toList)(s)
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

object PoissonDglmGibbsAncestor extends App with PoissonDglm with PoissonData {
  val n = 200
  val model = Dlm.Model(mod.f, mod.g)
  val initFilter = ParticleFilter.filter(model, data, params, n)
  val conditionedState = ParticleGibbs.sampleState(
    initFilter.map(d => d.state.map((d.time, _)).toList).toList, 
    initFilter.last.weights.toList
  ).draw

  val priorW = InverseGamma(11.0, 1.0)

  val mcmcStep = (s: LatentState, p: Dlm.Parameters) => for {
    w <- GibbsSampling.sampleSystemMatrix(priorW, model.g, s.toArray)
    (ll, state) <- ParticleGibbsAncestor.filter(n, params, model, data.toList)(s)
  } yield (state, Dlm.Parameters(p.v, w, p.m0, p.c0))

  val iters = MarkovChain((conditionedState, params)){ case (x, p) => mcmcStep(x, p) }.
    steps.
    map(_._2).
    take(10000)

  val headers = rfc.withHeader("W")
  def formatParameters(p: Dlm.Parameters) = {
    List(p.w.data(0))
  }

  Streaming.writeChain(formatParameters, 
    "data/poisson_dglm_gibbs_ancestor.csv", headers)(iters)
}
