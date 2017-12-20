# Fit a Poisson DGLM using Particle Gibbs using Newcastles HPC, Topsy
poisson_dglm_gibbs:
	sbt assembly
	cp target/scala-2.11/bayesian_dlms-assembly-0.2-SNAPSHOT.jar poisson_dglm.jar
	ssh topsy -t mkdir -p /share/nobackup/a9169110/poisson_dglm/data
	scp data/poisson_dglm.csv topsy:/share/nobackup/a9169110/poisson_dglm/data/.
	scp poisson_dglm.jar poisson_dglm.qsub topsy:/share/nobackup/a9169110/poisson_dglm/.
	ssh topsy -f "cd /share/nobackup/a9169110/poisson_dglm && qsub poisson_dglm.qsub"
	ssh topsy -t qstat

site: correlated second_order first_order seasonal knit_site

knit_site:
	sbt "tut"
	RScript -e 'Sys.setenv(RSTUDIO_PANDOC="/Applications/RStudio.app/Contents/MacOS/pandoc"); rmarkdown::render_site(input = "tut")'

correlated: simulate_correlated simulate_correlated_trend filter_correlated gibbs_correlated

simulate_correlated_trend:
	sbt "runMain dlm.examples.FirstOrderLinearTrendDlm"

simulate_correlated:
	sbt "runMain dlm.examples.SimulateCorrelated"

filter_correlated: data/correlated_dlm.csv
	sbt "runMain dlm.examples.FilterCorrelatedDlm"

gibbs_correlated: data/correlated_dlm.csv
	sbt "runMain dlm.examples.GibbsCorrelated"

second_order: simulate_second_order filter_second_order gibbs_second_order

simulate_second_order:
	sbt "runMain dlm.examples.SimulateSecondOrderDlm"

filter_second_order: data/second_order_dlm.csv
	sbt "runMain dlm.examples.FilterSecondOrderDlm"

gibbs_second_order: data/second_order_dlm.csv
	sbt "runMain dlm.examples.GibbsSecondOrder"

first_order: simulate_first_order filter_first_order smooth_first_order gibbs_first_order

simulate_first_order:
	sbt "runMain dlm.examples.SimulateDlm"

filter_first_order: data/first_order_dlm.csv
	sbt "runMain dlm.examples.FilterDlm"

smooth_first_order: data/first_order_dlm.csv
	sbt "runMain dlm.examples.SmoothDlm"

gibbs_first_order: data/first_order_dlm.csv
	sbt "runMain dlm.examples.GibbsParameters"

seasonal: simulate_seasonal smooth_seasonal filter_seasonal forecast_seasonal

simulate_seasonal:
	sbt "runMain dlm.examples.SimulateSeasonalDlm"

smooth_seasonal: data/seasonal_dlm.csv
	sbt "runMain dlm.examples.SmoothSeasonalDlm"

filter_seasonal: data/seasonal_dlm.csv
	sbt "runMain dlm.examples.FilterSeasonalDlm"

forecast_seasonal: data/seasonal_dlm.csv
	sbt "runMain dlm.examples.ForecastSeasonal"

# Fit a seasonal model using Newcastles HPC, Topsy
gibbs_seasonal:
	sbt assembly
	cp target/scala-2.11/bayesian_dlms-assembly-0.2-SNAPSHOT.jar seasonal_dlm.jar
	ssh topsy -t mkdir -p /share/nobackup/a9169110/seasonal_dlm/data
	scp data/seasonal_dlm.csv topsy:/share/nobackup/a9169110/seasonal_dlm/data/.
	scp seasonal_dlm.jar seasonal_dlm.qsub topsy:/share/nobackup/a9169110/seasonal_dlm/.
	ssh topsy -f "cd /share/nobackup/a9169110/seasonal_dlm && qsub seasonal_dlm.qsub"
	ssh topsy -t qstat


student_t: student_t_gibbs student_t_pmmh student_t_pg

simulate_student_t: 
	sbt "runMain dlm.examples.SimulateStudentT"

student_t_gibbs_local: data/student_t_dglm.csv
	sbt "runMain dlm.examples.StudentTGibbs"

student_t_gibbs: data/student_t_dglm.csv
	sbt assembly
	cp target/scala-2.11/bayesian_dlms-assembly-0.3.0-SNAPSHOT.jar student_t_gibbs.jar
	ssh topsy -t mkdir -p /share/nobackup/a9169110/student_t_gibbs/data
	scp data/student_t_dglm.csv topsy:/share/nobackup/a9169110/student_t_gibbs/data/.
	scp student_t_gibbs.jar student_t_gibbs.qsub topsy:/share/nobackup/a9169110/student_t_gibbs/.
	ssh topsy -f "cd /share/nobackup/a9169110/student_t_gibbs && qsub student_t_gibbs.qsub"
	ssh topsy -t qstat

student_t_pmmh: data/student_t_dglm.csv
	sbt assembly
	cp target/scala-2.11/bayesian_dlms-assembly-0.3.0-SNAPSHOT.jar student_t_pmmh.jar
	ssh topsy -t mkdir -p /share/nobackup/a9169110/student_t_pmmh/data
	scp data/student_t_dglm.csv topsy:/share/nobackup/a9169110/student_t_pmmh/data/.
	scp student_t_pmmh.jar student_t_pmmh.qsub topsy:/share/nobackup/a9169110/student_t_pmmh/.
	ssh topsy -f "cd /share/nobackup/a9169110/student_t_pmmh && qsub student_t_pmmh.qsub"
	ssh topsy -t qstat

student_t_pg: data/student_t_dglm.csv
	sbt assembly
	cp target/scala-2.11/bayesian_dlms-assembly-0.3.0-SNAPSHOT.jar student_t_pg.jar
	ssh topsy -t mkdir -p /share/nobackup/a9169110/student_t_pg/data
	scp data/student_t_dglm.csv topsy:/share/nobackup/a9169110/student_t_pg/data/.
	scp student_t_pg.jar student_t_pg.qsub topsy:/share/nobackup/a9169110/student_t_pg/.
	ssh topsy -f "cd /share/nobackup/a9169110/student_t_pg && qsub student_t_pg.qsub"
	ssh topsy -t qstat

get_student_t_data: 
	scp topsy:/share/nobackup/a9169110/student_t_pg/data/*.csv data/.
	scp topsy:/share/nobackup/a9169110/student_t_pmmh/data/*.csv data/.
	scp topsy:/share/nobackup/a9169110/student_t_gibbs/data/*.csv data/.
