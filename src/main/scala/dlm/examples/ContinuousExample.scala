package dlm.examples

import dlm.model._
import breeze.linalg._
import breeze.stats.distributions._
import cats._
import cats.implicits._
import java.nio.file.Paths
import kantan.csv._
import kantan.csv.ops._

trait SeasonalContMod {
  // combine models
  val mod = Dlm.polynomial(1) |+| Dlm.seasonal(24, 3) |+| Dlm.seasonal(24 * 7, 3)

  // combine continuous time G matrices
  val g = (dt: Double) =>
    List(
      DenseMatrix((1.0)),
      ContinuousTime.seasonalG(24, 3)(dt),
      ContinuousTime.seasonalG(24 * 7, 3)(dt)
    ).reduce(Dlm.blockDiagonal)
  
  // convert the model to continuous time
  val contMod = ContinuousTime.dlm2Model(mod, g)

  val p = Dlm.Parameters(
    v = DenseMatrix((1.0)),
    w = diag(DenseVector(0.5, 1.0, 0.2, 0.6, 0.5, 
      0.125, 0.2, 0.6, 0.5, 0.125, 0.2, 0.6, 0.5)),
    m0 = DenseVector.zeros[Double](13),
    c0 = DenseMatrix.eye[Double](13))
}

trait SeasonalIrregData {
  val rawData = Paths.get("data/seasonal_dlm_irregular.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader(false))
  val observations = reader.
    collect { 
      case Success(a) => Data(a.head, Some(a(1)).map(DenseVector(_)))
    }.
    toArray
}

object SimulateContinuous extends App with SeasonalContMod {
  // generate a list of irregular times
  val times = (1 to 1000).map(_.toDouble).
    filter(_ => Uniform(0, 1).draw < 0.8)

  val sims = ContinuousTime.simulate(times, contMod, p)

  val out = new java.io.File("data/seasonal_dlm_irregular.csv")

  def format(f: (Data, DenseVector[Double])): List[Double] = {
    List(f._1.time) ++ (f._1.observation.get.data ++ f._2.data).toList
  }

  val headers = rfc.withHeader(false)

  out.writeCsv(sims.map(format), headers)
}

object FilterSeasonalIrregular extends App with SeasonalContMod with SeasonalIrregData {
 
  val filtered = ExactFilter.filter(contMod, observations, p)

  val out = new java.io.File("data/seasonal_dlm_irregular_filtered.csv")

  def formatFiltered(f: KalmanFilter.State): List[Double] = {
    List(f.time) ++ (DenseVector.vertcat(f.mt, diag(f.ct)).data).toList
  }

  val headers = rfc.withHeader(false)

  out.writeCsv(filtered.map(formatFiltered), headers)
}

object BackwardSample extends App with SeasonalContMod with SeasonalIrregData {
  val states = Monad[Rand].replicateA(100, ExactBackSample.ffbs(contMod, observations, p)).
    draw.
    flatten.
    groupBy(_._1).
    map { case (time, state) => 
      state.
        reduce((a, b) => (a._1, a._2 + b._2)).
        map(_ / 100.0) 
    }

  val out = new java.io.File("data/seasonal_dlm_irregular_back_sample.csv")

  def format(f: (Time, DenseVector[Double])): List[Double] = {
    List(f._1) ++ f._2.data.toList
  }

  val headers = rfc.withHeader(false)

  out.writeCsv(states.map(format), headers)
}

object GibbsSampleIrregular extends App with SeasonalContMod with SeasonalIrregData {
  def priorW(w: DenseMatrix[Double]) =
    diag(p.v).map(InverseGamma(21.0, 10.0).logPdf).sum

  // full MCMC step for V and W
  val mcmcStep = (s: Dlm.Parameters) => for {
    state <- ExactBackSample.ffbs(contMod, observations, p)
    w <- GibbsSampling.sampleSystemMatrixCont(
      InverseGamma(4.0, 5.0), contMod.g, state)
    v <- GibbsSampling.sampleObservationMatrix(
      InverseGamma(4.0, 5.0), contMod.f, state, observations)
  } yield Dlm.Parameters(v, w, p.m0, p.c0)

  val iters = MarkovChain(p)(mcmcStep).
    steps.
    take(10000)

  val headers = rfc.withHeader("V", "W1", "W2", "W3", "W4", 
    "W5", "W6", "W7", "W8", "W9", "W10", "W11", "W12", "W13")

  def formatParameters(p: Dlm.Parameters) = {
    DenseVector.vertcat(diag(p.v), diag(p.w)).data.toList
  }

  Streaming.writeChain(formatParameters, "data/seasonal_irregular_gibbs.csv", headers)(iters)
}
