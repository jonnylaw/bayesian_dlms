package benchmark.bench

import org.openjdk.jmh.annotations.{Benchmark, State, Scope}
import core.dlm.model._
import cats.implicits._

object ResamplingBenchmark {

  @State(Scope.Benchmark)
  class Weights {
    val n = 50
    val x = Vector.fill(n)(1)
    val ws = Vector.fill(n)(1.0 / n)
  }
}

class ResamplingBenchmark {
  import ResamplingBenchmark._

  @Benchmark
  def metropolisResampling(x: Weights) = {
    ParticleFilter.metropolisResampling(10)(x.x, x.ws)
  }

  @Benchmark
  def multinomialResample(x: Weights) = {
    ParticleFilter.multinomialResample(x.x, x.ws)
  }
}
