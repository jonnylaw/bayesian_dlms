// import dlm.model._
// import breeze.linalg.{DenseMatrix, DenseVector, cond, diag}
// import breeze.stats.distributions.{ChiSquared, Gamma, MarkovChain}
// import breeze.stats.covmat
// import breeze.stats.{meanAndVariance, variance, mean}
// import org.scalatest._
// import prop._
// import org.scalacheck.Gen
// import org.scalactic.Equality

// // write some more tests for the Gibbs thing
// class GibbsTest extends PropSpec with GeneratorDrivenPropertyChecks with Matchers with BreezeGenerators {
//   val w = symmetricPosDefMatrix(2, 100)
//   val mod = Dlm.polynomial(2)
//   val state = w map (Dlm.simulateState(mod, _).steps.take(100).toArray)

//   property("Difference squared should produce a vector of the correct length") {
//     forAll(state) { (x: Array[(Time, DenseVector[Double])]) => 
//       // add whenever clause in here
//       val n = x.size
//       val diff = GibbsSampling.stateSquaredDifference(mod, x)

//       val mean = diff / (n - 1.0)

//       assert(diff.size === 2)
//       assert(mean === 1.0 +- 0.1)
//     }
//   }
// }
