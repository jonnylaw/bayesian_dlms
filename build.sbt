scalaVersion in ThisBuild := "2.11.11"

name := "bayesian_dlms"

organization := "com.github.jonnylaw"

version := "0.2-SNAPSHOT"

libraryDependencies ++= Seq(
  "org.scalanlp"        %% "breeze"             % "0.13.2",
  "org.scalanlp"        %% "breeze-natives"     % "0.13.2",
  "org.typelevel"       %% "cats-core"          % "0.9.0",
  "com.nrinaudo"        %% "kantan.csv-cats"    % "0.2.1",
  "io.spray"            %%  "spray-json"        % "1.3.3",
  "org.scalatest"       %% "scalatest"          % "3.0.4"  % "test"
)

publishMavenStyle := true

publishTo := {
  val nexus = "https://oss.sonatype.org/"
  if (isSnapshot.value)
    Some("snapshots" at nexus + "content/repositories/snapshots") 
  else
    Some("releases"  at nexus + "service/local/staging/deploy/maven2")
}

credentials += Credentials(Path.userHome / ".sbt" / ".credentials")

licenses := Seq("APL2" -> url("http://www.apache.org/licenses/LICENSE-2.0.txt"))

homepage := Some(url("https://jonnylaw.github.io/bayesian_dlms"))

scmInfo := Some(
  ScmInfo(
    url("https://github.com/jonnylaw/bayesian_dlms"),
    "scm:git@github.com:jonnylaw/bayesian_dlms.git"
  )
)

developers := List(
  Developer(id="1", name="Jonny Law", email="law.jonny@googlemail.com", url=url("https://jonnylaw.github.io/blog"))
)
