package examples.dlm

import dlm.core.model._
import cats.implicits._
import Dlm._
import breeze.linalg._
import java.nio.file.Paths
import kantan.csv._
import kantan.csv.ops._
import akka.actor.ActorSystem
import akka.stream._
import scaladsl._

trait CorrelatedModel {
  // first define two models, one for each time series
  val mod1 = polynomial(1)
  val mod2 = polynomial(1)

  // combine the models in an outer product
  val model = mod1 |*| mod2

  // specify the parameters for the joint model
  val v = diag(DenseVector(1.0, 4.0))
  val w = DenseMatrix((0.75, 0.5), (0.5, 1.25))
  val c0 = DenseMatrix.eye[Double](2)

  val p = DlmParameters(v, w, DenseVector.zeros[Double](2), c0)
}

trait CorrelatedData {
  val rawData = Paths.get("examples/data/correlated_dlm.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => Data(a.head, DenseVector(a(1).some, a(2).some))
  }.toVector
}

object SimulateCorrelated extends App with CorrelatedModel {
  val sims = Dlm.simulateRegular(model, p, 0.1).steps.take(1000)

  val out = new java.io.File("examples/data/correlated_dlm.csv")
  val headers = rfc.withHeader(
    Seq("time") ++
      Seq.range(1, 3).map(i => s"observation_$i") ++
      Seq.range(1, 3).map(i => s"state_$i"): _*)
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

object FilterCorrelatedDlm
    extends App
    with CorrelatedModel
    with CorrelatedData {
  val filtered =
    SvdFilter(SvdFilter.advanceState(p, model.g)).filter(model, data, p)

  val out = new java.io.File("examples/data/correlated_dlm_filtered.csv")

  def formatFiltered(f: SvdState) = {
    val ct = f.uc * diag(f.dc) * f.uc.t
    f.time :: DenseVector.vertcat(f.mt, diag(ct)).data.toList
  }

  val headers = rfc.withHeader(false)

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object GibbsCorrelated extends App with CorrelatedModel with CorrelatedData {
  implicit val system = ActorSystem("gibbs_correlated")
  implicit val materializer = ActorMaterializer()

  val iters = GibbsWishart
    .sample(model,
            InverseGamma(6.0, 5.0),
            InverseWishart(4.0, DenseMatrix.eye[Double](2)),
            p,
            data)

  def formatParameters(s: GibbsSampling.State) = {
    (s.p.v.data ++ s.p.w.data).toList
  }

  // write iters to file
  Streaming
    .writeParallelChain(iters,
                        2,
                        24000,
                        "examples/data/correlated_dlm_gibbs",
                        formatParameters)
    .runWith(Sink.onComplete(_ => system.terminate()))
}

object FirstOrderLinearTrendDlm extends App {
  val mod1 = polynomial(1)
  val mod2 = polynomial(2)

  val composedModel = Dlm.outerSumModel(mod1, mod2)

  val p = DlmParameters(
    v = diag(DenseVector(1.0, 2.0)),
    w = diag(DenseVector(2.0, 3.0, 1.0)),
    m0 = DenseVector.zeros[Double](3),
    c0 = DenseMatrix.eye[Double](3)
  )

  val sims = Dlm.simulateRegular(composedModel, p, 1.0).steps.take(1000)

  val out = new java.io.File("examples/data/first_order_and_linear_trend.csv")
  val headers = rfc.withHeader("time",
                               "observation_1",
                               "observation_2",
                               "state_1",
                               "state_2",
                               "state_3")
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

object SusteInvestment extends App with CorrelatedModel {
  implicit val system = ActorSystem("sutse_investment")
  implicit val materializer = ActorMaterializer()

  val rawData = Paths.get("examples/data/invest2.dat")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader(false))
  val data = reader
    .collect {
      case Right(a) => Data(0.0, DenseVector(a(0).some, a(1).some))
    }
    .toVector
    .zipWithIndex
    .map {
      case (d, i) =>
        d.copy(time = i + 1960.0)
    }

  def alpha(a: Double, b: Double) = {
    (2 * b + a * a) / b
  }

  def beta(a: Double, b: Double) = {
    (a / b) * (a * a + b)
  }

  val meanV = 0.1
  val variance = 1000.0
  val meanW = 1.0

  val priorV = InverseGamma(alpha(meanV, variance), beta(meanV, variance))
  val priorW = InverseGamma(alpha(meanW, variance), beta(meanW, variance))

  val initP = DlmParameters(
    v = diag(DenseVector.fill(2)(priorV.draw)),
    w = diag(DenseVector.fill(2)(priorW.draw)),
    m0 = DenseVector.zeros[Double](2),
    c0 = DenseMatrix.eye[Double](2)
  )

  val iters = GibbsSampling
    .sample(model, priorV, priorW, initP, data)

  def formatParameters(s: GibbsSampling.State) = {
    DenseVector.vertcat(diag(s.p.v), DenseVector(s.p.w.data)).data.toList
  }

  Streaming
    .writeParallelChain(iters,
                        2,
                        24000,
                        "examples/data/correlated_investment",
                        formatParameters)
    .runWith(Sink.onComplete(_ => system.terminate()))
}
