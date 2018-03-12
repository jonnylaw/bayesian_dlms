site: correlated second_order first_order seasonal knit_site

knit_site:
	sbt "tut"
	RScript -e 'Sys.setenv(RSTUDIO_PANDOC="/Applications/RStudio.app/Contents/MacOS/pandoc"); rmarkdown::render_site(input = "tut")'

clean_site:
	rm -rf tut
	rm -rf docs
	mkdir tut
	mkdir docs

poisson_dglm:
	sbt "runMain dlm.examples.SimulatePoissonDglm"
	sbt "runMain dlm.examples.PoissonDglmGibbs"
	sbt "runMain dlm.examples.PoissonDglmGibbsAncestor"

correlated:
	sbt "runMain dlm.examples.FirstOrderLinearTrendDlm"
	sbt "runMain dlm.examples.SimulateCorrelated"
	sbt "runMain dlm.examples.FilterCorrelatedDlm"
	sbt "runMain dlm.examples.GibbsCorrelated"

second_order: 
	sbt "runMain dlm.examples.SimulateSecondOrderDlm"
	sbt "runMain dlm.examples.FilterSecondOrderDlm"
	sbt "runMain dlm.examples.GibbsSecondOrder"

first_order: 
	sbt "runMain dlm.examples.SimulateDlm"
	sbt "runMain dlm.examples.FilterDlm"
	sbt "runMain dlm.examples.SmoothDlm"
	sbt "runMain dlm.examples.GibbsParameters"

seasonal: 
	sbt "runMain dlm.examples.SimulateSeasonalDlm"
	sbt "runMain dlm.examples.SmoothSeasonalDlm"
	sbt "runMain dlm.examples.FilterSeasonalDlm"
	sbt "runMain dlm.examples.ForecastSeasonal"
	sbt "runMain dlm.examples.SeasonalGibbsSampling"

student_t:
	sbt "runMain dlm.examples.SimulateStudentT"
	sbt "runMain dlm.examples.StudentTGibbs"
	sbt "runMain dlm.examples.StudentTpmmh"
	sbt "runMain dlm.examples.StudentTPG"
