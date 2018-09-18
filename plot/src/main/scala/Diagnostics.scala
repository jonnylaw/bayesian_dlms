package plot

import com.cibo.evilplot.plot._
import com.cibo.evilplot.plot.aesthetics.DefaultTheme._
import com.cibo.evilplot.numeric.Point
import breeze.stats.mean

object Diagnostics {
  /**
    * Calculate the autocorrelation of a sequence of data for a given lag
    * @param xs a time series of Doubles
    * @param lag the lag to calculate the autocorrelation function at
    * @return the autocorrelation at a given lag
    */
  def acf(xs: Vector[Double], lag: Int): Double = {
    val average = mean(xs)
    val n = xs.size
    val laggedSum = (xs.take(n-lag) zip xs.drop(lag)).
      map { case (x, xl) => (x - average) * (xl - average) }.
      sum
    val centeredSum = xs.map(x => x - average).sum

    laggedSum / centeredSum
  }

  /**
    * Plot the autocorrelation of a single parameter
    */
  def autocorrelation(xs: Vector[Double]): Plot = {
    val acs = (1 to 30).map(l => acf(xs, l))
    BarChart(acs)
      .xAxis()
      .yAxis()
      .frame()
  }

  /**
    * 
    */
  def autocorrelations(xs: Vector[Vector[Double]]): Plot = {
    Facets(Vector(xs.transpose.map(autocorrelation)))
  }

  /**
    * Plot a traceplot for a single parameter
    */
  def traceplot(xs: Vector[Double]): Plot = {
    LinePlot(xs.zipWithIndex.map { case (x, i) => Point(i, x) })
      .xAxis()
      .yAxis()
      .frame()
  }

  def traceplots(xs: Vector[Vector[Double]]): Plot = {
    Facets(Vector(xs.transpose.map(traceplot)))
  }

  def histogram(xs: Vector[Double]): Plot = {
    Histogram(xs)
      .xAxis()
      .yAxis()
      .frame()
  }

  def histograms(xs: Vector[Vector[Double]]): Plot = {
    Facets(Vector(xs.transpose.map(histogram)))
  }

  def density(xs: Vector[Vector[Double]]): Plot = ???

  /**
    * Plot Traceplot and histograms for an MCMC chain
    */
  def diagnostics(xs: Vector[Vector[Double]]): Plot = {
    val t = xs.transpose
    Facets(Vector(t.map(traceplot), t.map(histogram)))
  }
}
