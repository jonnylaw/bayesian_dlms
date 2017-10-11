scalaVersion := "2.11.11"

libraryDependencies ++= Seq(
  "org.scalanlp"   %% "breeze"             % "0.13.2",
  "org.scalanlp"   %% "breeze-natives"     % "0.13.2",
  "org.typelevel"  %% "cats-core"          % "0.9.0",
  "com.nrinaudo"   %% "kantan.csv-cats"    % "0.2.1",
  "io.spray"       %%  "spray-json"        % "1.3.3",
  "com.github.fommil.netlib" % "all" % "1.1.2",
  "org.scalacheck" %% "scalacheck"         % "1.13.4" % "test"
)
