# bayesian_dlms
Bayesian Inference for Dynamic Linear Models (DLMs)

# Install

Using sbt:

```scala
libraryDependencies += "com.github.jonnylaw" % "bayesian_dlm" % "0.1"
```

Check out the [documentation](https://jonnylaw.github.io/bayesian_dlms/).

# Learning More About DLMs

Dynamic Linear Models (DLMs) are [state space models](https://en.wikipedia.org/wiki/State-space_representation) where the latent-state and observation models are linear and Gaussian. DLMs are used to model time series data and the distribution of the latent state can be found exactly using the [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter) for sequential online estimation and the Kalman Smoother for offline estimation.

To read more about DLMs I recommend the following textbooks:

Petris, Giovanni, Sonia Petrone, and Patrizia Campagnoli. "Dynamic Linear Models with R." [Springer, 2009](http://www.springer.com/gb/book/9780387772370)

Harrison, Jeff, and Mike West. "Bayesian forecasting & dynamic models." [Springer, 1999](http://www.springer.com/gb/book/9780387947259)
