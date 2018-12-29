all: correlated first_order second_order seasonal simulate_linear_trend gibbs_investment ar1 ou dlm_fsv site

site:
	cd .. && sbt ";project core; tut"
	cd ../tut && Rscript -e "rmarkdown::render_site()"

ar1:
	cd .. && sbt ";project examples; runMain com.github.jonnylaw.dlm.example.SimulateArDlm; runMain com.github.jonnylaw.dlm.example.FilterArDlm; runMain com.github.jonnylaw.dlm.example.ParametersAr"

ou:
	cd .. && sbt ";project examples; runMain com.github.jonnylaw.dlm.example.SimulateOuDlm; runMain com.github.jonnylaw.dlm.example.FilterOuDlm; runMain com.github.jonnylaw.dlm.example.FitOuDlm"

correlated:
	cd .. && sbt ";project examples; runMain com.github.jonnylaw.dlm.example.SimulateCorrelated; runMain com.github.jonnylaw.dlm.example.FilterCorrelatedDlm; runMain com.github.jonnylaw.dlm.example.FirstOrderLinearTrendDlm; runMain com.github.jonnylaw.dlm.example.GibbsCorrelated"

second_order:
	cd .. && sbt ";project examples; runMain com.github.jonnylaw.dlm.example.SimulateSecondOrderDlm; runMain com.github.jonnylaw.dlm.example.FilterSecondOrderDlm; runMain com.github.jonnylaw.dlm.example.GibbsSecondOrder"

gibbs_investment:
	cd .. && sbt ";project examples; runMain com.github.jonnylaw.dlm.example.GibbsInvestParameters"

first_order:
	cd .. && sbt ";project examples; runMain com.github.jonnylaw.dlm.example.SimulateDlm; runMain com.github.jonnylaw.dlm.example.SmoothDlm; runMain com.github.jonnylaw.dlm.example.FilterDlm; runMain com.github.jonnylaw.dlm.example.GibbsParameters; runMain com.github.jonnylaw.dlm.example.SampleStates"

seasonal:
	cd .. && sbt ";project examples; runMain com.github.jonnylaw.dlm.example.SimulateSeasonalDlm; runMain com.github.jonnylaw.dlm.example.SampleStates; runMain com.github.jonnylaw.dlm.example.FilterSeasonalDlm; runMain com.github.jonnylaw.dlm.example.SmoothSeasonalDlm; runMain com.github.jonnylaw.dlm.example.SeasonalGibbsSampling"

online:
	cd .. && sbt ";project examples; runMain com.github.jonnylaw.dlm.example.ConjFilter; runMain com.github.jonnylaw.dlm.example.LiuAndWest; runMain com.github.jonnylaw.dlm.example.Storvik; runMain com.github.jonnylaw.dlm.example.RbFilter"

dlm_fsv:
	cd .. && sbt ";project examples; runMain com.github.jonnylaw.dlm.example.SimulateDlmFsvSystem; runMain com.github.jonnylaw.dlm.example.FitDlmFsvSystem"

clean:
	rm ../examples/data/*.csv
