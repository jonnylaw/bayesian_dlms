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
import breeze.linalg.{DenseMatrix, DenseVector, diag}

object Streaming {
  /**
    * Write a single chain
    * @param iters the iterations of the MCMC chain
    * @param formatParameters a function to properly format the parameters
    * to write them to a file
    * @param file the file to write the parameters to
    * @param config additional configuration as CsvConfiguration
    * from the kantan CSV package
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
    * Parse parameters from a collection containing the diagonal
    * entries of the v, w matrices
    */
  def parseDiagonalParameters(
    vDim: Int, wDim: Int)(ps: Vector[Double]): DlmParameters = {

    val v = ps.take(vDim).map(_.toDouble).toArray
    val w = ps.drop(vDim).take(wDim).map(_.toDouble).toArray
    val m0 = ps.drop(vDim + wDim).take(wDim).map(_.toDouble).toArray
    val c0 = ps.drop(vDim + 2 * wDim).take(wDim * wDim).
      map(_.toDouble).toArray

    DlmParameters(
      diag(DenseVector(v)),
      diag(DenseVector(w)),
      DenseVector(m0),
      new DenseMatrix(wDim, wDim, c0))
  }


  /**
    * Read an MCMC Chain into a list of doubles
    */
  def readMcmcChain(filename: String) = {
    val mcmcChain = Paths.get(filename)
    mcmcChain.asCsvReader[Vector[Double]](rfc.withHeader)
  }

  /**
    * Calculate the column means of List of List of doubles
    */
  def colMeans(params: List[List[Double]]): List[Double] = {
    params.transpose.map(a => breeze.stats.mean(a))
  }

  def quantile[A: Ordering](xs: Seq[A], prob: Double): A = {
    val index = math.floor(xs.length * prob).toInt
    val ordered = xs.sorted
    ordered(index)
  }

  /**
    * Streaming mean
    */
  def mean = Flow[Double].
    fold((0.0, 1.0)){ case ((avg, n), b) =>
      ((avg * n + b) / (n + 1), n + 1)
    }.
    map(_._1)

  def meanDlmFsvParameters(vDim: Int, wDim: Int, p: Int, k: Int) =
    Flow[DlmFsvParameters].
      fold((DlmFsvParameters.empty(vDim, wDim, p, k), 1.0)){ case ((avg, n), b) => 
        (avg.map(_ * n).add(b).map(_  / (n + 1)), n + 1)
      }.
      map(_._1)

  def meanDlmFsvSystemParameters(vDim: Int, wDim: Int, k: Int) =
    Flow[DlmFsvParameters].
      fold((DlmFsvSystem.emptyParams(vDim, wDim, k), 1.0)){ case ((avg, n), b) => 
        (avg.map(_ * n).add(b).map(_  / (n + 1)), n + 1)
      }.
      map(_._1)

  /**
    * Calculate the streaming mean of DLM parameters
    */
  def meanParameters(vDim: Int, wDim: Int) = {
    Flow[DlmParameters].
      fold((DlmParameters.empty(vDim, wDim), 1.0))((acc, b) => {
        val (avg: DlmParameters, n: Double) = acc
        (avg.map(_ * n).add(b).map(_  / (n + 1)), n + 1)
      }).
      map(_._1)
  }

  def meanSvParameters = Flow[SvParameters].
    fold((SvParameters.empty, 1.0)){(acc, b) =>
           val (avg: SvParameters, n: Double) = acc
           (avg.map(_ * n).add(b).map(_  / (n + 1)), n + 1)
         }.
    map(_._1)

  def meanFsvParameters(p: Int, k: Int) = Flow[FsvParameters].
    fold((FsvParameters.empty(p, k), 1.0)){(acc, b) =>
      val (avg: FsvParameters, n: Double) = acc
      (avg.map(_ * n).add(b).map(_  / (n + 1)), n + 1)
    }.
    map(_._1)

  /**
    * Create an Akka stream from a Markov Chain
    */
  def streamChain[A](chain: breeze.stats.distributions.Process[A],
                     nIters: Int): Source[A, NotUsed] = {

    Source.fromIterator(() => chain.steps.take(nIters))
  }

  def writeChainSink[A](
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
        runWith(writeChainSink(s"${filename}_$i.csv", format))
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
