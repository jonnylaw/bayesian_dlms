scalaVersion in ThisBuild := "2.12.1"

name := "bayesian_dlm"

organization := "com.github.jonnylaw"

version := "0.1"

libraryDependencies ++= Seq(
  "org.scalanlp"        %% "breeze"             % "0.13.2",
  "org.scalanlp"        %% "breeze-natives"     % "0.13.2",
  "org.typelevel"       %% "cats-core"          % "0.9.0",
  "com.nrinaudo"        %% "kantan.csv-cats"    % "0.2.1",
  "io.spray"            %%  "spray-json"        % "1.3.3",
  "org.scalatest"       %% "scalatest"          % "3.0.4"  % "test"
)

libraryDependencies += "com.lihaoyi" % "ammonite" % "1.0.2" % "test" cross CrossVersion.full

sourceGenerators in Test += Def.task {
  val file = (sourceManaged in Test).value / "amm.scala"
  IO.write(file, """object amm extends App { ammonite.Main().run() }""")
  Seq(file)
}.taskValue
