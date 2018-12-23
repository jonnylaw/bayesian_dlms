package plot

import com.cibo.evilplot.plot._
import breeze.linalg.DenseVector
import com.cibo.evilplot.numeric.Point
import com.cibo.evilplot.plot.aesthetics.DefaultTheme._

object TimeSeries {

  /**
    * Line plot with states
    */
  def plotWithState(
      xs: Vector[(Double, DenseVector[Option[Double]], DenseVector[Double])])
    : Unit = ???

  /**
    * Convert a vector of Data into a vector of vector of points which can each be plotted
    */
  def listOfObservations(xs: Vector[(Double, DenseVector[Option[Double]])])
    : Vector[Vector[Point]] = {

    val times: Vector[Double] = xs.map(_._1)
    val states: Vector[Vector[Option[Double]]] =
      xs.map(_._2.data.toVector).transpose

    states.map(x => observationToPoint(times zip x))
  }

  def observationToPoint(xs: Vector[(Double, Option[Double])]): Vector[Point] =
    xs.map { case (t, yo) => yo map (y => Point(t, y)) }.flatten

  /**
    * Plot a single time series
    */
  def plotObservation(xs: Vector[Point]) = {
    LinePlot(xs)
      .xAxis()
      .yAxis()
      .frame()
  }

  /**
    * Line plot of time against observations
    */
  def plotObservations(
      xs: Vector[(Double, DenseVector[Option[Double]])]): Plot = {
    val plots = for {
      obs <- listOfObservations(xs)
    } yield plotObservation(obs)

    Facets(Vector(plots).transpose)
  }
}
