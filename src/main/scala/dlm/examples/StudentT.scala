package dlm.examples

import dlm.model._
import breeze.stats.distributions._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._

trait StudenttDglm {
  val dlm = Dlm.polynomial(1)
  val mod = Dglm.studentT(3, dlm)
  val params = Dlm.Parameters(
    DenseMatrix(3.0), 
    DenseMatrix(0.1), 
    DenseVector(0.0), 
    DenseMatrix(1.0))
}

trait StudenttData {
  val rawData = Paths.get("data/student_t_dglm.csv")
  val reader = rawData.asCsvReader[(Time, Double, Double)](rfc.withHeader)
  val data = reader.
    collect { 
      case Success(a) => Data(a._1, Some(a._2).map(DenseVector(_)))
    }.
    toArray
}

object SimulateStudentT extends App with StudenttDglm {
  val sims = Dglm.simulate(mod, params).
    steps.
    take(1000)

  val out = new java.io.File("data/student_t_dglm.csv")
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
  * Use Kalman Filtering to determine the parameters of the Student's t-distribution DGLM
  */
object StudentTGibbs extends App with StudenttDglm with StudenttData {
  val iters = GibbsSampling.studentT(3, data, 
    InverseGamma(2.0, 10.0), mod, params).
    steps.
    take(100000).
    map(_.p)

  def formatParameters(p: Dlm.Parameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  val headers = rfc.withHeader("scale", "W")
  Streaming.writeChain(formatParameters, 
    "data/student_t_dglm_gibbs.csv", headers)(iters)
}

object StudentTpmmh extends App with StudenttDglm with StudenttData {
  def prior(p: Dlm.Parameters) = {
    InverseGamma(5.0, 4.0).logPdf(p.w(0, 0)) +
    InverseGamma(5.0, 4.0).logPdf(p.v(0, 0))
  }

  val n = 500
  val iters = Metropolis.dglm(mod, data,
    Metropolis.symmetricProposal(0.01), prior, params, n).
    steps.
    take(100000).
    map(_.parameters)

  def formatParameters(p: Dlm.Parameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  val headers = rfc.withHeader("scale", "W")
  Streaming.writeChain(formatParameters, 
    "data/student_t_dglm_pmmh.csv", headers)(iters)
}

object StudentTPG extends App with StudenttDglm with StudenttData {
  val n = 200
  val model = Dlm.Model(mod.f, mod.g)
  val initFilter = ParticleFilter.filter(model, data, params, n)
  val conditionedState = ParticleGibbs.sampleState(
    initFilter.map(d => d.state.map((d.time, _)).toList).toList, 
    initFilter.last.weights.toList
  ).draw

  val priorW = InverseGamma(11.0, 100.0)
  def priorV(v: DenseMatrix[Double]) = {
    InverseGamma(11.0, 30.0).logPdf(v(0,0))
  }

  case class State(
    s:  LatentState,
    p:  Dlm.Parameters,
    ll: Double
  )

  def mcmcStep(state: State) = for {
    newW <- GibbsSampling.sampleSystemMatrix(priorW, model.g, state.s.toArray)
    propV <- Metropolis.proposeDiagonalMatrix(0.01)(state.p.v)
    (ll, latentState) <- ParticleGibbsAncestor.filter(n, params.copy(v = propV), 
      model, data.toList)(state.s)
    a = ll + priorV(propV) - state.ll
    u <- Uniform(0, 1)
    next = if (math.log(u) < a) {
      (ll + priorV(propV), propV)
    } else {
      (state.ll, state.p.v)
    }
  } yield State(latentState, state.p.copy(v = next._2, w = newW), next._1)

  val initState = State(conditionedState, params, -1e99)
  val iters = MarkovChain(initState)(mcmcStep).
    steps.
    map(_.p).
    take(100000)

  val headers = rfc.withHeader("scale", "W")
  def formatParameters(p: Dlm.Parameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  Streaming.writeChain(formatParameters, 
    "data/student_t_dglm_gibbs_ancestor.csv", headers)(iters)
}
