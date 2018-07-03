simulations: correlated second_order seasonal student_t ar1 poisson_dglm

knit_site:
	sbt "core/tut"
	RScript -e 'Sys.setenv(RSTUDIO_PANDOC="/Applications/RStudio.app/Contents/MacOS/pandoc"); rmarkdown::render_site(input = "tut")'

clean_site:
	rm -rf tut
	rm -rf docs
	mkdir tut
	mkdir docs

poisson_dglm:
	sbt ";examples/runMain core.dlm.examples.SimulatePoissonDglm; examples/runMain core.dlm.examples.PoissonDglmGibbs"

correlated:
	mkdir -p data
	sbt ";examples/runMain core.dlm.examples.FirstOrderLinearTrendDlm; examples/runMain core.dlm.examples.SimulateCorrelated; examples/runMain core.dlm.examples.FilterCorrelatedDlm; examples/runMain core.dlm.examples.GibbsCorrelated"

second_order: 
	sbt ";examples/runMain core.dlm.examples.SimulateSecondOrderDlm; examples/runMain core.dlm.examples.FilterSecondOrderDlm; examples/runMain core.dlm.examples.GibbsSecondOrder; examples/runMain core.dlm.examples.GibbsInvestParameters"

first_order: 
	sbt ";examples/runMain core.dlm.examples.SimulateDlm; examples/runMain core.dlm.examples.FilterDlm; examples/runMain core.dlm.examples.SmoothDlm; examples/runMain core.dlm.examples.SampleStates; examples/runMain core.dlm.examples.GibbsParameters"

seasonal: 
	sbt ";examples/runMain core.dlm.examples.SimulateSeasonalDlm; examples/runMain core.dlm.examples.SmoothSeasonalDlm; examples/runMain core.dlm.examples.FilterSeasonalDlm; examples/runMain core.dlm.examples.SeasonalGibbsSampling; examples/runMain core.dlm.examples.ForecastSeasonal"

student_t:
	sbt ";examples/runMain core.dlm.examples.SimulateStudentT; examples/runMain core.dlm.examples.StudentTGibbs; examples/runMain core.dlm.examples.StudentTpmmh;examples/runMain core.dlm.examples.StudentTPG"

ar1:
	sbt ";examples/runMain core.dlm.examples.SimulateArDlm; examples/runMain core.dlm.examples.FilterArDlm; examples/runMain core.dlm.examples.ParametersAr"

filtering:
	sbt ";examples/runMain core.dlm.examples.FilterArDlm; examples/runMain core.dlm.examples.FilterCorrelatedDlm; ;examples/runMain core.dlm.examples.FilterDlm; examples/runMain core.dlm.examples.FilterSeasonalDlm; examples/runMain core.dlm.examples.FilterSecondOrderDlm"
