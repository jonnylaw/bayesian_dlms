package benchmark.bench

import org.openjdk.jmh.annotations.{Benchmark, State, Scope}
import com.github.jonnylaw.dlm._
import breeze.linalg._

object FfbsBenchmark {

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

class FfbsBenchmark {
  import FfbsBenchmark._

  @Benchmark
  def naiveFfbs(mod: ModelState) = {
    Smoothing.ffbsDlm(mod.model, mod.data, mod.p)
  }

  @Benchmark
  def svdFfbs(mod: ModelState) = {
    SvdSampler.ffbsDlm(mod.model, mod.data, mod.p)
  }

  // @Benchmark
  // def arrayFfbs(mod: ModelState) = {
  //   FilterArray.ffbsNaive(mod.model, mod.data, mod.p)
  // }

  // @Benchmark
  // def arrayFfbsSvd(mod: ModelState) = {
  //   FilterArray.ffbsSvd(mod.model, mod.data, mod.p)
  // }
}
