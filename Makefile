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
	sbt ";examples/runMain examples.dlm.SimulatePoissonDglm; examples/runMain examples.dlm.PoissonDglmGibbs"

correlated:
	mkdir -p data
	sbt ";examples/runMain examples.dlm.FirstOrderLinearTrendDlm; examples/runMain examples.dlm.SimulateCorrelated; examples/runMain examples.dlm.FilterCorrelatedDlm; examples/runMain examples.dlm.GibbsCorrelated"

second_order: 
	sbt ";examples/runMain examples.dlm.SimulateSecondOrderDlm; examples/runMain examples.dlm.FilterSecondOrderDlm; examples/runMain examples.dlm.GibbsSecondOrder; examples/runMain examples.dlm.GibbsInvestParameters"

first_order: 
	sbt ";examples/runMain examples.dlm.SimulateDlm; examples/runMain examples.dlm.FilterDlm; examples/runMain examples.dlm.SmoothDlm; examples/runMain examples.dlm.SampleStates; examples/runMain examples.dlm.GibbsParameters"

seasonal: 
	sbt ";examples/runMain examples.dlm.SimulateSeasonalDlm; examples/runMain examples.dlm.SmoothSeasonalDlm; examples/runMain examples.dlm.FilterSeasonalDlm; examples/runMain examples.dlm.SeasonalGibbsSampling; examples/runMain examples.dlm.ForecastSeasonal"

student_t:
	sbt ";examples/runMain examples.dlm.SimulateStudentT; examples/runMain examples.dlm.StudentTGibbs; examples/runMain examples.dlm.StudentTpmmh;examples/runMain examples.dlm.StudentTPG"

ar1:
	sbt ";examples/runMain examples.dlm.SimulateArDlm; examples/runMain examples.dlm.FilterArDlm; examples/runMain examples.dlm.ParametersAr"

filtering:
	sbt ";examples/runMain examples.dlm.FilterArDlm; examples/runMain examples.dlm.FilterCorrelatedDlm; ;examples/runMain examples.dlm.FilterDlm; examples/runMain examples.dlm.FilterSeasonalDlm; examples/runMain examples.dlm.FilterSecondOrderDlm"

stochastic_volatility:
	sbt ";examples/runMain examples.dlm.SimulateSv ;examples/runMain examples.dlm.FitSv; examples/runMain examples.dlm.SimulateFsv; examples/runMain examples.dlm.FitFsv"

