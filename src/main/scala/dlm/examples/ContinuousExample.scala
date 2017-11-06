package dlm.examples

import dlm.model._
import breeze.linalg._
import breeze.stats.distributions._
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

object FilterSeasonalIrregular extends App with SeasonalContMod {
  val rawData = Paths.get("data/seasonal_dlm_irregular.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader(false))
  val observations = reader.
    collect { 
      case Success(a) => Data(a.head, Some(a(1)).map(DenseVector(_)))
    }.
    toArray

  val filtered = ExactFilter.filter(contMod, observations, p)

  val out = new java.io.File("data/seasonal_dlm_irregular_filtered.csv")

  def formatFiltered(f: KalmanFilter.State): List[Double] = {
    List(f.time) ++ (DenseVector.vertcat(f.mt, diag(f.ct)).data).toList
  }

  val headers = rfc.withHeader(false)

  out.writeCsv(filtered.map(formatFiltered), headers)
}

