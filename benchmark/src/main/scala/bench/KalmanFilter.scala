package benchmark.bench

import org.openjdk.jmh.annotations.{Benchmark, State, Scope}
import core.dlm.model._
import breeze.linalg._
import cats.implicits._

object KalmanFilterBenchmark {

  @State(Scope.Benchmark)
  class ModelState {
    val model = Dlm.polynomial(1)
    val p = DlmParameters(
      v = diag(DenseVector(3.0)),
      w = diag(DenseVector(1.0)),
      m0 = DenseVector.fill(1)(0.0),
      c0 = diag(DenseVector(1.0))
    )

    val data =
      Dlm.simulateRegular(model, p, 1.0).steps.take(10).toVector.map(_._1)
  }
}

class KalmanFilterBenchmark {
  import KalmanFilterBenchmark._

  @Benchmark
  def kalmanFilter(mod: ModelState) = {
    KalmanFilter.filter(mod.model, mod.data, mod.p)
  }

  // @Benchmark
  // def svdFilter(mod: ModelState) = {
  //   SvdFilter.filter(mod.model, mod.data, mod.p)
  // }

  // @Benchmark
  // def arrayFilterSvd(mod: ModelState) = {
  //   FilterArray.filterSvd(mod.model, mod.data, mod.p)
  // }

  // @Benchmark
  // def arrayFilter(mod: ModelState) = {
  //   FilterArray.filterNaive(mod.model, mod.data, mod.p)
  // }

  @Benchmark
  def genFilter(mod: ModelState) = {
    KalmanFilter.filter(mod.model, mod.data, mod.p)
  }
}
