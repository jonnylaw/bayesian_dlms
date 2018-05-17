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
	sbt ";core/runMain core.dlm.examples.SimulatePoissonDglm; core/runMain core.dlm.examples.PoissonDglmGibbs"

correlated:
	mkdir -p data
	sbt ";core/runMain core.dlm.examples.FirstOrderLinearTrendDlm; core/runMain core.dlm.examples.SimulateCorrelated; core/runMain core.dlm.examples.FilterCorrelatedDlm; core/runMain core.dlm.examples.GibbsCorrelated"

second_order: 
	sbt ";core/runMain core.dlm.examples.SimulateSecondOrderDlm; core/runMain core.dlm.examples.FilterSecondOrderDlm; core/runMain core.dlm.examples.GibbsSecondOrder; core/runMain core.dlm.examples.GibbsInvestParameters"

first_order: 
	sbt ";core/runMain core.dlm.examples.SimulateDlm; core/runMain core.dlm.examples.FilterDlm; core/runMain core.dlm.examples.SmoothDlm; core/runMain core.dlm.examples.SampleStates; core/runMain core.dlm.examples.GibbsParameters"

seasonal: 
	sbt ";core/runMain core.dlm.examples.SimulateSeasonalDlm; core/runMain core.dlm.examples.SmoothSeasonalDlm; core/runMain core.dlm.examples.FilterSeasonalDlm; core/runMain core.dlm.examples.SeasonalGibbsSampling; core/runMain core.dlm.examples.ForecastSeasonal"

student_t:
	sbt ";core/runMain core.dlm.examples.SimulateStudentT; core/runMain core.dlm.examples.StudentTGibbs; core/runMain core.dlm.examples.StudentTpmmh;core/runMain core.dlm.examples.StudentTPG"

ar1:
	sbt ";core/runMain core.dlm.examples.SimulateArDlm; core/runMain core.dlm.examples.FilterArDlm; core/runMain core.dlm.examples.ParametersAr"

filtering:
	sbt ";core/runMain core.dlm.examples.FilterArDlm; core/runMain core.dlm.examples.FilterCorrelatedDlm; ;core/runMain core.dlm.examples.FilterDlm; core/runMain core.dlm.examples.FilterSeasonalDlm; core/runMain core.dlm.examples.FilterSecondOrderDlm"
