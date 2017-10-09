correlated: simulate_correlated filter_correlated gibbs_correlated

simulate_correlated:
	cd .. && sbt "run-main dlm.examples.SimulateCorrelated"

filter_correlated: ../data/CorrelatedDlm.csv
	cd .. && sbt "run-main dlm.examples.FilterCorrelatedDlm"

gibbs_correlated: ../data/CorrelatedDlm.csv
	cd .. && sbt "run-main dlm.examples.GibbsCorrelated"

second_order: simulate_second_order filter_second_order gibbs_second_order

simulate_second_order:
	cd .. && sbt "run-main dlm.examples.SimulateSecondOrder"

filter_second_order: ../data/second_order_dlm.csv
	cd .. && sbt "run-main dlm.examples.FilterSecondOrder"

gibbs_second_order: ../data/second_order_dlm.csv
	cd .. && sbt "run-main dlm.examples.GibbsSecondOrder"

