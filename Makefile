simulations: correlated second_order seasonal student_t ar1 poisson_dglm

knit_site:
	sbt "tut"
	RScript -e 'Sys.setenv(RSTUDIO_PANDOC="/Applications/RStudio.app/Contents/MacOS/pandoc"); rmarkdown::render_site(input = "tut")'

clean_site:
	rm -rf tut
	rm -rf docs
	mkdir tut
	mkdir docs

poisson_dglm:
	sbt ";runMain dlm.examples.SimulatePoissonDglm; runMain dlm.examples.PoissonDglmGibbs; runMain dlm.examples.PoissonDglmGibbsAncestor"

correlated:
	mkdir -p data
	sbt ";runMain dlm.examples.FirstOrderLinearTrendDlm; runMain dlm.examples.SimulateCorrelated; runMain dlm.examples.FilterCorrelatedDlm; runMain dlm.examples.GibbsCorrelated"

second_order: 
	sbt ";runMain dlm.examples.SimulateSecondOrderDlm; runMain dlm.examples.FilterSecondOrderDlm; runMain dlm.examples.GibbsSecondOrder"

first_order: 
	sbt ";runMain dlm.examples.SimulateDlm; runMain dlm.examples.FilterDlm; runMain dlm.examples.SmoothDlm; runMain dlm.examples.SampleStates; runMain dlm.examples.GibbsParameters"

seasonal: 
	sbt ";runMain dlm.examples.SimulateSeasonalDlm; runMain dlm.examples.SmoothSeasonalDlm; runMain dlm.examples.FilterSeasonalDlm; runMain dlm.examples.ForecastSeasonal; runMain dlm.examples.SeasonalGibbsSampling"

student_t:
	sbt ";runMain dlm.examples.SimulateStudentT; runMain dlm.examples.StudentTGibbs; runMain dlm.examples.StudentTpmmh;runMain dlm.examples.StudentTPG"

ar1:
	sbt ";runMain dlm.examples.SimulateArDlm; runMain dlm.examples.FilterArDlm; runMain dlm.examples.ParametersAr"

filtering:
	sbt ";runMain dlm.examples.FilterArDlm; runMain dlm.examples.FilterCorrelatedDlm; ;runMain dlm.examples.FilterDlm; runMain dlm.examples.FilterSeasonalDlm; runMain dlm.examples.FilterSecondOrderDlm"
