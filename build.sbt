lazy val commonSettings = Seq(
  scalaVersion := "2.12.7",
  name := "bayesian_dlms",
  organization := "com.github.jonnylaw",
  version := "0.5.0",
  scalacOptions ++= Seq(
    "-encoding", "UTF-8",   // source files are in UTF-8
    "-deprecation",         // warn about use of deprecated APIs
    "-unchecked",           // warn about unchecked type parameters
    "-feature",             // warn about misused language features
    "-language:higherKinds",// allow higher kinded types without `import scala.language.higherKinds`
    "-Xlint",               // enable handy linter warnings
                            // "-Xfatal-warnings",     // turn compiler warnings into errors
    "-Ypartial-unification", // allow the compiler to unify type constructors of different arities
    "-language:implicitConversions" // allow implicit conversion of DLM -> DGLM
  ),
  crossScalaVersions := Seq("2.11.11","2.12.7"),
  credentials += Credentials(
    "Sonatype Nexus Repository Manager",
    "oss.sonatype.org",
    sys.env.getOrElse("SONATYPE_USER", ""),
    sys.env.getOrElse("SONATYPE_PASS", "")),
  licenses := Seq("APL2" -> url("http://www.apache.org/licenses/LICENSE-2.0.txt")),
  homepage := Some(url("https://jonnylaw.github.io/bayesian_dlms")),
  scmInfo := Some(
    ScmInfo(
      url("https://github.com/jonnylaw/bayesian_dlms"),
      "scm:git@github.com:jonnylaw/bayesian_dlms.git"
    )
  ),
  developers := List(
    Developer(id="1", name="Jonny Law", email="law.jonny@googlemail.com", url=url("https://jonnylaw.github.io/blog"))
  ),
  publishMavenStyle := true,
  publishTo := {
    val nexus = "https://oss.sonatype.org/"
    if (isSnapshot.value)
      Some("snapshots" at nexus + "content/repositories/snapshots")
    else
      Some("releases"  at nexus + "service/local/staging/deploy/maven2")
  },
  useGpg := false,
  usePgpKeyHex("AFB3D11B342D342A"),
  pgpPublicRing := baseDirectory.value / "project" / ".gnupg" / "pubring.gpg",
  pgpSecretRing := baseDirectory.value / "project" / ".gnupg" / "secring.gpg",
  pgpPassphrase := sys.env.get("PGP_PASS").map(_.toArray),
  git.baseVersion := "0.5.0",
  git.gitTagToVersionNumber := {
    case ReleaseTag(v) => Some(v)
    case _ => None
  },
  git.formattedShaVersion := {
    val suffix = git.makeUncommittedSignifierSuffix(git.gitUncommittedChanges.value, git.uncommittedSignifier.value)

    git.gitHeadCommit.value map { _.substring(0, 7) } map { sha =>
      git.baseVersion.value + "-" + sha + suffix
    }
  }
)

val ReleaseTag = """^v([\d\.]+)$""".r

// scalafmtOnCompile in ThisBuild := true

lazy val core = (project in file("core"))
  .settings(
    commonSettings,
    libraryDependencies ++= Seq(
      "org.scalanlp"        %% "breeze"             % "0.13.2",
      "org.scalanlp"        %% "breeze-natives"     % "0.13.2",
      "com.nrinaudo"        %% "kantan.csv-cats"    % "0.4.0",
      "com.nrinaudo"        %% "kantan.csv-java8"   % "0.4.0",
      "com.nrinaudo"        %% "kantan.csv-generic" % "0.4.0",
      "org.typelevel"       %% "cats-core"          % "1.5.0",
      "org.typelevel"       %% "cats-testkit"       % "1.5.0",
      "com.typesafe.akka"   %% "akka-stream"        % "2.5.9",
      "org.scalatest"       %% "scalatest"          % "3.0.5"  % "test",
      "org.scalacheck"      %% "scalacheck"         % "1.14.0" % "test"
    ),
    tutSourceDirectory := baseDirectory.value / "../R",
    tutTargetDirectory := baseDirectory.value / "../tut",
    tutNameFilter := ".+\\.Rmd".r,
  )
  .enablePlugins(TutPlugin)
  .enablePlugins(GitVersioning)

addCommandAlias("ci-all",  ";+clean ;+compile ;+test ;+package")
addCommandAlias("release", ";+publishSigned ;sonatypeReleaseAll")

lazy val benchmark = project
  .dependsOn(core)
  .enablePlugins(JmhPlugin)

lazy val plot = project
  .settings(
    resolvers += Resolver.bintrayRepo("cibotech", "public"),
    libraryDependencies ++= Seq(
      "org.scalanlp" %% "breeze"   % "0.13.2",
      "com.cibo"     %% "evilplot" % "0.3.2"
    )
  )

lazy val examples = project.
  settings(
    libraryDependencies ++= Seq(
      "com.stripe" %% "rainier-core" % "0.1.3",
      "com.stripe" %% "rainier-cats" % "0.1.3"
    )
  )
  .dependsOn(core, plot)

