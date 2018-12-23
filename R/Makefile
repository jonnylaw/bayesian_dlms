all: correlated first_order second_order seasonal simulate_linear_trend gibbs_investment ar1 ou

ar1:
	cd .. && sbt ";project examples; runMain examples.dlm.SimulateArDlm; runMain examples.dlm.FilterArDlm; runMain examples.dlm.ParametersAr"

ou:
	cd .. && sbt ";project examples; runMain examples.dlm.SimulateOuDlm; runMain examples.dlm.FilterOuDlm; runMain examples.dlm.FitOuDlm"

correlated:
	cd .. && sbt ";project examples; runMain examples.dlm.SimulateCorrelated; runMain examples.dlm.FilterCorrelatedDlm; runMain examples.dlm.FirstOrderLinearTrendDlm; runMain examples.dlm.GibbsCorrelated"

second_order:
	cd .. && sbt ";project examples; runMain examples.dlm.SimulateSecondOrderDlm; runMain examples.dlm.FilterSecondOrderDlm; runMain examples.dlm.GibbsSecondOrder"

gibbs_investment:
	cd .. && sbt ";project examples; runMain examples.dlm.GibbsInvestParameters"

first_order:
	cd .. && sbt ";project examples; runMain examples.dlm.SimulateDlm; runMain examples.dlm.SmoothDlm; runMain examples.dlm.FilterDlm; runMain examples.dlm.GibbsParameters; runMain examples.dlm.SampleStates"

seasonal:
	cd .. && sbt ";project examples; runMain examples.dlm.SimulateSeasonalDlm; runMain examples.dlm.SampleStates; runMain examples.dlm.FilterSeasonalDlm; runMain examples.dlm.SmoothSeasonalDlm; runMain examples.dlm.SeasonalGibbsSampling"

online:
	cd .. && sbt ";project examples; runMain examples.dlm.ConjFilter; runMain examples.dlm.LiuAndWest; runMain examples.dlm.Storvik; runMain examples.dlm.RbFilter"

clean:
	rm ../examples/data/*.csv
