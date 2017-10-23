# Fit a seasonal model using Newcastles HPC, Topsy
seasonal_model_simulated:
	sbt assembly
	cp target/scala-2.11/bayesiandlm-assembly-0.1-SNAPSHOT.jar seasonal_dlm.jar
	ssh topsy -t mkdir -p /share/nobackup/a9169110/seasonal_dlm/data
	scp data/seasonal_dlm.csv topsy:/share/nobackup/a9169110/seasonal_dlm/data/.
	scp seasonal_dlm.jar seasonal_dlm.qsub topsy:/share/nobackup/a9169110/seasonal_dlm/.
	ssh topsy -t "cd /share/nobackup/a9169110/seasonal_dlm && dos2unix seasonal_dlm.qsub"
	ssh topsy -f "cd /share/nobackup/a9169110/seasonal_dlm && qsub seasonal_dlm.qsub"
	ssh topsy -t qstat

site: correlated second_order first_order seasonal

correlated: simulate_correlated simulate_correlated_trend filter_correlated gibbs_correlated

simulate_correlated_trend:
	sbt "run-main dlm.examples.FirstOrderLinearTrendDlm"

simulate_correlated:
	sbt "run-main dlm.examples.SimulateCorrelated"

filter_correlated: data/correlated_dlm.csv
	sbt "run-main dlm.examples.FilterCorrelatedDlm"

gibbs_correlated: data/correlated_dlm.csv
	sbt "run-main dlm.examples.GibbsCorrelated"

second_order: simulate_second_order filter_second_order gibbs_second_order

simulate_second_order:
	sbt "run-main dlm.examples.SimulateSecondOrderDlm"

filter_second_order: data/second_order_dlm.csv
	sbt "run-main dlm.examples.FilterSecondOrderDlm"

gibbs_second_order: data/second_order_dlm.csv
	sbt "run-main dlm.examples.GibbsSecondOrder"

first_order: simulate_first_order filter_first_order smooth_first_order gibbs_first_order

simulate_first_order:
	sbt "run-main dlm.examples.SimulateDlm"

filter_first_order: data/first_order_dlm.csv
	sbt "run-main dlm.examples.FilterDlm"

smooth_first_order: data/first_order_dlm.csv
	sbt "run-main dlm.examples.SmoothDlm"

gibbs_first_order: data/first_order_dlm.csv
	sbt "run-main dlm.examples.GibbsParameters"

seasonal: simulate_seasonal seasonal_sample_state smooth_seasonal filter_seasonal

simulate_seasonal:
	sbt "run-main dlm.examples.SimulateSeasonalDlm"

seasonal_sample_state: data/seasonal_dlm.csv
	sbt "run-main dlm.examples.SampleStates"

smooth_seasonal: data/seasonal_dlm.csv
	sbt "run-main dlm.examples.SmoothSeasonalDlm"

filter_seasonal: data/seasonal_dlm.csv
	sbt "run-main dlm.examples.FilterSeasonalDlm"
