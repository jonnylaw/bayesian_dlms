# A first order joint model
first_order_joint:
	sbt assembly
	cp target/scala-2.11/bayesiandlm-assembly-0.1-SNAPSHOT.jar first_order_joint_dlm.jar
	scp first_order_joint_dlm.jar bessel:/data/a9169110/.
	scp data/humidity_temperature_1114.csv bessel:/data/a9169110/data/.
	ssh bessel -f screen -S joint_dlm -dm cd /data/a9169110 && jdk/bin/java -cp first_order_joint_dlm.jar dlm.examples.urbanobservatory.JointFirstOrder 
	ssh bessel -t screen -ls

# A first order regression DLM
# with intercept
regression_dlm:
	sbt assembly
	cp target/scala-2.11/bayesiandlm-assembly-0.1-SNAPSHOT.jar regression_dlm.jar
	scp regression_dlm.jar bessel:/data/a9169110/.
	scp data/humidity_temperature_1114.csv bessel:/data/a9169110/data/.
	ssh bessel -f screen -S RegressionDlm -dm cd /data/a9169110 && jdk/bin/java -cp regression_dlm.jar dlm.examples.urbanobservatory.RegressionDlm
	ssh bessel -t screen -ls

regression_topsy:
	sbt assembly
	cp target/scala-2.11/bayesiandlm-assembly-0.1-SNAPSHOT.jar regression_dlm.jar
	scp regression_dlm.jar regression_dlm.qsub topsy:/share/nobackup/a9169110/.
	ssh topsy -f qsub regression_dlm.qsub

seasonal_temperature_model:
	sbt assembly
	cp target/scala-2.11/bayesiandlm-assembly-0.1-SNAPSHOT.jar seasonal_dlm.jar
	scp seasonal_dlm.jar bessel:/data/a9169110/.
	ssh bessel -f screen -S TemperatureGibbs -dm cd /data/a9169110 && jdk/bin/java -cp seasonal_dlm.jar dlm.examples.urbanobservatory.FitTemperatureModel
	ssh bessel -t screen -ls

correlated_seasonal_model:
	sbt assembly
	cp target/scala-2.11/bayesiandlm-assembly-0.1-SNAPSHOT.jar correlated_seasonal_dlm.jar
	scp correlated_seasonal_dlm.jar bessel:/data/a9169110/.
	ssh bessel -f screen -S CorrelatedSeasonalGibbs -dm "cd /data/a9169110 && jdk/bin/java -cp correlated_seasonal_dlm.jar dlm.examples.urbanobservatory.JointModel"
	ssh bessel -t screen -ls
