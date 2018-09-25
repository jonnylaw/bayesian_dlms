package dlm.core.model

import java.io.File
import java.nio.file.Paths
import kantan.csv._
import kantan.csv.ops._
import scala.concurrent.Future
import akka.stream._
import akka.stream.scaladsl._
import akka.NotUsed
import akka.util.ByteString

object Streaming {
  /**
    * Write a single chain
    * @param iters the iterations of the MCMC chain
    * @param formatParameters a function to properly format the parameters
    * to write them to a file
    * @param file the file to write the parameters to
    * @param config additional configuration as CsvConfiguration
    * from the kantan CSV package
    * @return a Monix Task for writing an iterator to a file
    */
  def writeChain(
      formatParameters: DlmParameters => List[Double],
      filename: String,
      config: CsvConfiguration
  )(iters: Iterator[DlmParameters]) = {
    val file = new File(filename)
    val writer = file.asCsvWriter[List[Double]](config)

    // write iters to file
    while (iters.hasNext) {
      writer.write(formatParameters(iters.next))
    }

    writer.close()
  }

  /**
    * Read an MCMC Chain into a list of doubles
    */
  def readMcmcChain(filename: String) = {
    val mcmcChain = Paths.get("data/seasonal_dlm_gibbs.csv")
    mcmcChain.asCsvReader[List[Double]](rfc.withHeader)
  }

  /**
    * Calculate the column means of List of List of doubles
    */
  def colMeans(params: List[List[Double]]): List[Double] = {
    params.transpose.map(a => breeze.stats.mean(a))
  }

  /**
    * Create an Akka stream from a Markov Chain
    */
  def streamChain[A](chain: breeze.stats.distributions.Process[A],
                     nIters: Int): Source[A, NotUsed] = {

    Source.fromIterator(() => chain.steps.take(nIters))
  }

  def writeChain[A](
      filename: String,
      format: A => List[Double]
  ): Sink[A, Future[IOResult]] = {

    Flow[A]
      .map(a => ByteString(format(a).mkString(", ") + "\n"))
      .toMat(FileIO.toPath(Paths.get(filename)))(Keep.right)
  }

  /**
    * Given a single MCMC, write different realisations using
    * the same initial parameters
    * @param nChains the number of parallel chains
    * @param nIters the number of iterations
    * @param filename the prefix of the filename
    * @param format a function to format each row of the CSV output
    */
  def writeParallelChain[A](
    chain: breeze.stats.distributions.Process[A],
    nChains: Int,
    nIters: Int,
    filename: String,
    format: A => List[Double])(implicit m: Materializer) = {

    Source((0 until nChains)).mapAsync(nChains) { i =>
      streamChain(chain, nIters).
        runWith(writeChain(s"${filename}_$i.csv", format))
    }
  }

  /**
    * Filter a Stream to select only every nth iteration
    * @param xs an iterator
    * @param n the index of the iterator to select
    */
  def thinChain[A](n: Int) = {
    Flow[A].zipWithIndex.
      filter { case (a, i) => i % n == 0 }.
      map(_._1)
  }

  def readCsv[S](file: String) = {
    FileIO.fromPath(Paths.get(file)).
      via(Framing.delimiter(ByteString("\n"), 
        maximumFrameLength = 8192, allowTruncation = true)).
      map(_.utf8String).
      map(a => a.split(",").toVector)
  }

}
