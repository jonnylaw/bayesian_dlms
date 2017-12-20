![build status](https://travis-ci.org/jonnylaw/bayesian_dlms.svg?branch=master)

# bayesian_dlms
Bayesian Inference for Dynamic Linear Models (DLMs)

# Install

This package is crossbuilt for Scala 2.11.11 and Scala 2.12.1. To install using sbt add this to your `build.sbt`:

```scala
libraryDependencies += "com.github.jonnylaw" %% "bayesian_dlms" % "0.3.0"
```

Check out the [documentation](https://jonnylaw.github.io/bayesian_dlms/).

# Learning More About DLMs

Dynamic Linear Models (DLMs) are [state space models](https://en.wikipedia.org/wiki/State-space_representation) where the latent-state and observation models are linear and Gaussian. DLMs are used to model time series data and the distribution of the latent state can be found exactly using the [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter) for sequential online estimation and the Kalman Smoother for offline estimation.

To read more about DLMs I recommend the following textbooks:

Petris, Giovanni, Sonia Petrone, and Patrizia Campagnoli. "Dynamic Linear Models with R." [Springer, 2009](http://www.springer.com/gb/book/9780387772370)

Harrison, Jeff, and Mike West. "Bayesian forecasting & dynamic models." [Springer, 1999](http://www.springer.com/gb/book/9780387947259)

# Features  

* Building univariate and multivariate D(G)LMs
* Support for irregularly observed and explicitly missing data
* Kalman Filter
* Backwards Smoothing
* Forward-Filtering Backward Sampling (FFBS)
* Gibbs Sampling using d-Inverse Gamma Modelling
* Gibbs Sampling using the Inverse Wishart Distribution
* Metropolis Hastings Sampling
* Bootstrap Particle Filter for DGLMs
* Particle Gibbs Sampling
* Particle Gibbs with Ancestor Sampling

# Upcoming Features

* Matrix Normal Model
* Partial Observations of Multivariate Time Series
