package benchmark.bench

import org.openjdk.jmh.annotations.{Benchmark, State, Scope}
import dlm.core.model._
import breeze.linalg._
import cats.implicits._

object ArrayFilterBenchmark {

  @State(Scope.Benchmark)
  class ModelState {
    val model = Dlm.polynomial(1)
    val p = DlmParameters(
      v = diag(DenseVector(3.0)),
      w = diag(DenseVector(1.0)),
      m0 = DenseVector.fill(1)(0.0),
      c0 = diag(DenseVector(1.0))
    )

    val data = Dlm.simulateRegular(model, p, 1.0).
        steps.
        take(10).
        toVector.
        map(_._1)

    val ys = data.map(d => (d.time, d.observation(0)))
  }
}

class ArrayFilterBenchmark {
  import ArrayFilterBenchmark._

  @Benchmark
  def kalmanFilter(mod: ModelState) = {
    KalmanFilter.filterDlm(mod.model, mod.data, mod.p)
  }

  // @Benchmark
  // def arrayFilter(mod: ModelState) = {
  //   KalmanFilter(KalmanFilter.advanceState(mod.p, mod.model.g)).filterArray(mod.model, mod.data, mod.p)
  // }

  @Benchmark
  def univariateFilter(mod: ModelState) = {
    KalmanFilter.univariateKf(mod.ys, mod.p, mod.model)
  }
}
