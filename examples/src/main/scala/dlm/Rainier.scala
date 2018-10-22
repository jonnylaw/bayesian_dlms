package examples.dlm

import dlm.core.model._
import java.nio.file.Paths
import cats._
import cats.implicits._
import kantan.csv._
import kantan.csv.ops._
import breeze.linalg.{DenseMatrix, DenseVector}
import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._
import com.stripe.rainier.cats._

object RainierSv extends App {
  implicit val rng = ScalaRNG(1234) // set a seed

  val rawData = Paths.get("examples/data/sv_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader)
  val data = reader.collect {
    case Right(a) => (a.head, a(1))
  }.toVector.
    take(500)

  case class Parameters(phi: Real, mu: Real, sigma: Real)

  val priorSv: RandomVariable[(Parameters, Real)] = for {
    phi <- Beta(2, 5).param
    mu <- Normal(0, 1).param
    sigma <- Gamma(0, 1).param
    initVar = (sigma * sigma) / (1.0 - phi * phi)
    a0 <- Normal(mu, initVar.pow(0.5)).param
  } yield (Parameters(phi, mu, sigma), a0)

  def addTimePointSv(
    params: RandomVariable[(Parameters, Real)],
    y: Double): RandomVariable[(Parameters, Real)] =
    for {
      (p, a) <- params
      a1 <- Normal(p.mu + p.phi * (a - p.mu), p.sigma).param
      _ <- Normal(0, (a1 * 0.5).exp).fit(y)
    } yield (p, a1)

  val fullModel = data.map(_._2).foldLeft(priorSv)(addTimePointSv)

  val model = for {
    p <- fullModel
  } yield Map("mu" -> p._1.mu, "phi" -> p._1.phi, "sigma" -> p._1.sigma)

  val iters = model.sample(HMC(5), 1000, 1000)

  val out = new java.io.File("examples/data/sv_rainier.csv")
  val headers = rfc.withHeader("mu", "phi", "sigma")

  out.writeCsv(iters.map(_.values), headers)
}

object FactorSvRainier extends App {
  val p = 6
  val k = 1
  val rawData = Paths.get("examples/data/fsv_sims.csv")
  val reader = rawData.asCsvReader[List[Double]](rfc.withHeader(false))
  val data = reader.
    collect { 
      case Right(a) => a.drop(1).take(p).toVector
    }.
    toVector.
    take(500)

  implicit val rng = ScalaRNG(1234) // set a seed

  case class SvParameters(phi: Real, mu: Real, sigma: Real)

  case class FsvParameters(
    beta: Vector[Real],
    v:    Vector[Real],
    sv:   Vector[SvParameters])

  val priorSv: RandomVariable[(SvParameters, Real)] = for {
    phi <- Beta(2, 5).param
    mu <- Normal(0, 1).param
    sigma <- Gamma(0, 1).param
    initVar = (sigma * sigma) / (1.0 - phi * phi)
    a0 <- Normal(mu, initVar.pow(0.5)).param
  } yield (SvParameters(phi, mu, sigma), a0)

  val prior: RandomVariable[(FsvParameters, Vector[Real])] = for {
    sv <- Vector.fill(k)(priorSv).sequence
    beta <- Vector.
      fill(p * k - ((k / 2) * (k + 1)))(Normal(0, 5).param).sequence
    v <- Vector.fill(p)(Gamma(0, 1).param).sequence
  } yield (FsvParameters(beta, v, sv.map(_._1)), sv.map(_._2))

  def dot(a: Vector[Real], b: Vector[Real]): Real = 
    (a zip b).map { case (x, y) => x * y }.reduce(_ + _)

  // multiply beta by factors
  def betaFactor(
    p: Int,
    k: Int,
    beta: Vector[Real],
    fs: Vector[Real]): Vector[Real] = {
    val b: Vector[Vector[Real]] = Vector.tabulate(p, k){ (i, j) =>
      if (i == j) {
        1.0
      } else if (i > j) {
        beta(i + 2 * j)
      } else {
        0.0
      }
    }

    b.transpose.map(bj => dot(bj, fs))
  }

  def vectorToMap(v: Vector[Real], name: String): Map[String, Real] = {
    v.zipWithIndex.
      map { case (x, i) => Map(s"name_$i" -> x) }.
      foldLeft(Map[String, Real]())(_ ++ _)
  }

  def addTimePoint(
    params: RandomVariable[(FsvParameters, Vector[Real])],
    ys:     Vector[Double]): RandomVariable[(FsvParameters, Vector[Real])] =
    for {
      (ps, as) <- params
      a1 <- (ps.sv zip as).
        map { case (x, a) => Normal(x.mu + x.phi * (a - x.mu), x.sigma).param }.
        sequence
      f1 <- a1.map(a => Normal(0, (a * 0.5).exp).param).sequence
      mean = betaFactor(p, k, ps.beta, f1)
      _ <- (mean, ps.v, ys).zipped.
        map { case (m, v, y) => Normal(m, v).fit(y) }.
        sequence
    } yield (ps, a1)

  val fullModel = data.foldLeft(prior)(addTimePoint)

  val model = for {
    (p, at) <- fullModel
  } yield vectorToMap(p.beta, "beta") ++ vectorToMap(p.v, "v") ++
  p.sv.zipWithIndex.
    flatMap { case (ps, i) =>
      Map(s"phi_$i" -> ps.phi, s"mu_$i" -> ps.mu, s"sigma_$i" -> ps.sigma)}.
    foldLeft(Map[String, Real]())(_ + _)

  val iters = model.sample(Walkers(20), 10000, 10000, 10)

  val out = new java.io.File("examples/data/fsv_rainier.csv")
  val headers = rfc.withHeader(false)

  out.writeCsv(iters.map(_.values), headers)
}
