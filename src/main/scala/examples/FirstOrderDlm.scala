import Dlm._
import breeze.linalg.{DenseMatrix, DenseVector}
import java.io.{File, PrintWriter}
import math.log
import MetropolisHastings._

trait FirstOrderDlm {
  val mod = Model(f = (t: Time) => DenseMatrix((1.0)), g = (t: Time) => DenseMatrix((1.0)))
  val p = Parameters(Vector(log(0.5)), Vector(log(2.0)), Vector(0.0), Vector(1.0))
}

object SimulateDlm extends App with FirstOrderDlm {
  val data = simulate(1, mod, p).
    steps.
    take(300).
    toArray

  // serialise to JSON
  val pw = new PrintWriter(new File("data/FirstOrderDlm.csv"))

  val strings = data.
    map { case (d, x) => s"${d.time}, ${d.observation.get.data.mkString(", ")}, ${x.data.mkString(", ")}" }

  pw.write(strings.mkString("\n"))
  pw.close()    
}

object FilterDlm extends App with FirstOrderDlm {
  // read and deserialise JSON
  val data = scala.io.Source.fromFile("data/FirstOrderDlm.csv").
    getLines.
    map(_.split(",")).
    map(x => Data(x(0).toInt, Some(DenseVector(x(1).toDouble)))).
    toArray

  val filtered = KalmanFilter.kalmanFilter(mod, data, p)

  val pw = new PrintWriter(new File("data/FirstOrderDlmFiltered.csv"))
  pw.write(filtered.mkString("\n"))
  pw.close()
}

object DetermineParameters extends App with FirstOrderDlm {
  val data: Array[Data] = scala.io.Source.fromFile("data/FirstOrderDlm.csv").
    getLines.
    map(_.split(",")).
    map(x => Data(x(0).toInt, Some(DenseVector(x(1).toDouble)))).
    toArray

  val iters = metropolisHastingsDlm(mod, data, symmetricProposal(0.35), MhState(p, 0.0, 0)).
    steps.
    take(10000)

  // write iters to file
  val pw = new PrintWriter(new File("data/FirstOrderDlmIters.csv"))
  while (iters.hasNext) {
    pw.write(iters.next.toString + "\n")
  }
  pw.close()
}
