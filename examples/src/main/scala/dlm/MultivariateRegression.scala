package examples.dlm

import core.dlm.model._
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions._
import java.nio.file.Paths
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import akka.actor.ActorSystem
import akka.stream._
import scaladsl._

trait MultivariateRegression {
  // Build a multivariate regression DLM
  def model(x: Vector[Array[Double]]) =
    x.map(xi => Dlm.regression(xi.map(DenseVector(_)))).
      reduce(_ |*| _)

  val params = DlmParameters(
    v = diag(DenseVector.fill(4)(1.0)),
    w = diag(DenseVector.fill(8)(1.0)),
    m0 = DenseVector.fill(8)(0.0),
    c0 = diag(DenseVector.fill(8)(10.0))
  )
}

object UnivariateRegression extends App {
  val n = 300
  val xs = Array.iterate(0.0, n)(x => x + Uniform(-1, 1).draw).
    map(x => DenseVector(x))
  val params = DlmParameters(
    v = diag(DenseVector.fill(1)(1.0)),
    w = diag(DenseVector.fill(2)(1.0)),
    m0 = DenseVector.fill(2)(0.0),
    c0 = diag(DenseVector.fill(2)(10.0))
  )

  val sims = Dlm.simulateRegular(Dlm.regression(xs), params, 1.0).
    steps.
    take(n).
    toVector
  
  // write to file
  val out = new java.io.File("examples/data/univariate_regression_sims.csv")
  val colnames = Seq("time", "observation") ++
    (1 to 2).map(i => s"state_$i") ++ Seq("covariate")

  val headers = rfc.withHeader(colnames: _*)

  def formatData(d: ((Dlm.Data, DenseVector[Double]), DenseVector[Double])) = d match {
    case ((Dlm.Data(t, y), s), x) =>
      t :: KalmanFilter.flattenObs(y).data.toList ::: s.data.toList ::: x.data.toList
  }

  out.writeCsv(sims.zip(xs).map(formatData), headers)
}

/**
  * Simulate a multivariate regression model with 4 observations
  */
object SimulateMultivariateRegression extends App with MultivariateRegression {
  // simulate the covariates, p series of length n
  val p = 4
  val n = 300
  val xs = Vector.fill(p)(Array.iterate(0.0, n)(x => x + Uniform(-1, 1).draw))
 
  val sims = Dlm.simulateRegular(model(xs), params, 1.0).
    steps.
    take(300).
    toVector
  
  // write to file
  val out = new java.io.File("examples/data/multivariate_regression_sims.csv")
  val colnames = Seq("time") ++ (1 to 4).map(i => s"observation_$i") ++
    (1 to 8).map(i => s"state_$i") ++ (1 to 4).map(i => s"covariate_$i")
  val headers = rfc.withHeader(colnames: _*)

  def formatData(d: ((Dlm.Data, DenseVector[Double]), Vector[Double])) = d match {
    case ((Dlm.Data(t, y), s), x) =>
      t :: KalmanFilter.flattenObs(y).data.toList ::: s.data.toList ::: x.toList
  }

  out.writeCsv(sims.zip(xs.transpose).map(formatData), headers)
}

object FitMultivariateRegression extends App with MultivariateRegression {
  implicit val system = ActorSystem("multivariate_regression")
  implicit val materializer = ActorMaterializer()

  val rawData = Paths.get("examples/data/multivariate_regression_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => (Dlm.Data(a.head, DenseVector(a.drop(1).take(4).toArray).map(_.some)),
      a.drop(13).toArray)
  }.toVector

  val priorV = InverseGamma(4.0, 6.0)
  val priorW = InverseGamma(6.0, 15.0)

  val xs = data.map(_._2).transpose
  val iters = GibbsSampling.sample(model(xs.map(_.toArray)), priorV, priorW, params, data.map(_._1))

  def formatParameters(s: GibbsSampling.State) = {
    DenseVector.vertcat(diag(s.p.v), diag(s.p.w)).data.toList
  }

  Streaming.writeParallelChain(
    iters, 2, 10000, "examples/data/multivariate_regression_gibbs", formatParameters).
    runWith(Sink.onComplete(_ => system.terminate()))
}
