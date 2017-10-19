all: correlated first_order second_order seasonal simulate_linear_trend

correlated: simulate_correlated filter_correlated gibbs_correlated

simulate_correlated:
	cd .. && sbt "runMain dlm.examples.SimulateCorrelated"

filter_correlated: ../data/correlated_dlm.csv
	cd .. && sbt "runMain dlm.examples.FilterCorrelatedDlm"

gibbs_correlated: ../data/correlated_dlm.csv
	cd .. && sbt "runMain dlm.examples.GibbsCorrelated"

simulate_linear_trend:
	cd .. && sbt "runMain dlm.examples.FirstOrderLinearTrendDlm"

second_order: simulate_second_order filter_second_order gibbs_second_order

simulate_second_order:
	cd .. && sbt "runMain dlm.examples.SimulateSecondOrderDlm"

filter_second_order: ../data/second_order_dlm.csv
	cd .. && sbt "runMain dlm.examples.FilterSecondOrderDlm"

gibbs_second_order: ../data/second_order_dlm.csv
	cd .. && sbt "runMain dlm.examples.GibbsSecondOrder"

gibbs_investment: ../data/invest2.dat
	cd .. && sbt "runMain dlm.examples.GibbsInvestParameters"

first_order: simulate_first_order filter_first_order smooth_first_order gibbs_first_order

simulate_first_order:
	cd .. && sbt "runMain dlm.examples.SimulateDlm"

smooth_first_order: ../data/first_order_dlm.csv
	cd .. && sbt "runMain dlm.examples.SmoothDlm"

filter_first_order: ../data/first_order_dlm.csv
	cd .. && sbt "runMain dlm.examples.FilterDlm"

gibbs_first_order: ../data/first_order_dlm.csv
	cd .. && sbt "runMain dlm.examples.GibbsParameters"

seasonal: simulate_seasonal filter_seasonal sample_state_seasonal smooth_seasonal

simulate_seasonal:
	cd .. && sbt "runMain dlm.examples.SimulateSeasonalDlm"

sample_state_seasonal:
	cd .. && sbt "runMain dlm.examples.SampleStates"

filter_seasonal: ../data/seasonal_dlm.csv
	cd .. && sbt "runMain dlm.examples.FilterSeasonalDlm"

smooth_seasonal: ../data/seasonal_dlm.csv
	cd .. && sbt "runMain dlm.examples.SmoothSeasonalDlm"

clean:
	rm ../data/*.csv
