scalaVersion := "2.11.11"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.13.2",
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  "org.typelevel" %% "cats-core" % "1.0.0-MF",
  "org.scala-saddle" %% "saddle-core" % "1.3.+",
  "io.spray" %%  "spray-json" % "1.3.3",
  "org.scalacheck" %% "scalacheck" % "1.13.4" % "test"
)
