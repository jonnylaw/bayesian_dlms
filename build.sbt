scalaVersion in ThisBuild := "2.11.11"

name := "bayesian_dlms"

organization := "com.github.jonnylaw"

version := "0.3.0-SNAPSHOT"

scalacOptions ++= Seq(
  "-encoding", "UTF-8",   // source files are in UTF-8
  "-deprecation",         // warn about use of deprecated APIs
  "-unchecked",           // warn about unchecked type parameters
  "-feature",             // warn about misused language features
  "-language:higherKinds",// allow higher kinded types without `import scala.language.higherKinds`
  "-Xlint",               // enable handy linter warnings
//  "-Xfatal-warnings",     // turn compiler warnings into errors
  "-Ypartial-unification", // allow the compiler to unify type constructors of different arities
  "-language:implicitConversions" // allow implicit conversion of DLM -> DGLM
)

libraryDependencies ++= Seq(
  "org.scalanlp"        %% "breeze"             % "0.13.2",
  "org.scalanlp"        %% "breeze-natives"     % "0.13.2",
  "org.typelevel"        %% "cats-core"         % "1.0.0-RC1",
  "com.nrinaudo"        %% "kantan.csv-cats"    % "0.3.0",
  "org.scalatest"       %% "scalatest"          % "3.0.4"  % "test"
)

publishMavenStyle := true

crossScalaVersions := Seq("2.11.11","2.12.1")

// Enable Tut for typechecking and running scala documentation
enablePlugins(TutPlugin)
tutSourceDirectory := baseDirectory.value / "R"
tutTargetDirectory := baseDirectory.value / "tut"
tutNameFilter := ".+\\.Rmd".r

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
