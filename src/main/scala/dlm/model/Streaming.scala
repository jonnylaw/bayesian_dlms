package dlm.model

import java.io.File
import Dlm.Parameters
import breeze.stats.distributions.Process
import kantan.csv._
import kantan.csv.ops._

/**
  * Utility class for parallelism and IO
  */
object Streaming {
  /**
    * Write a single chain
    * @param iters the iterations of the MCMC chain
    * @param formatParameters a function to properly format the parameters to write them to a file
    * @param file the file to write the parameters to
    * @param config additional configuration as CsvConfiguration from the kantan CSV package
    * @return a Monix Task for writing an iterator to a file
    */
  def writeChain(
    formatParameters: Parameters => List[Double],
    filename: String,
    config: CsvConfiguration
  )(iters: Iterator[Parameters]) = {
    val file = new File(filename)
    val writer = file.asCsvWriter[List[Double]](config)

    // write iters to file
    while (iters.hasNext) {
      writer.write(formatParameters(iters.next))
    }

    writer.close()
  }
}
