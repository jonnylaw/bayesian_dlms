# A first order joint model
first_order_joint:
	sbt assembly
	cp target/scala-2.11/bayesiandlm-assembly-0.1-SNAPSHOT.jar first_order_joint_dlm.jar
	scp first_order_joint_dlm.jar bessel:/data/a9169110/.
	scp data/humidity_temperature_1114.csv bessel:/data/a9169110/data/.
	ssh bessel -f "cd /data/a9169110 && screen -S joint_dlm -dm jdk/bin/java -cp first_order_joint_dlm.jar dlm.examples.urbanobservatory.JointFirstOrder"
	ssh bessel -t screen -ls

# A first order regression DLM
# with intercept
regression_dlm:
	sbt assembly
	cp target/scala-2.11/bayesiandlm-assembly-0.1-SNAPSHOT.jar regression_dlm.jar
	scp regression_dlm.jar bessel:/data/a9169110/.
	scp data/humidity_temperature_1114.csv bessel:/data/a9169110/data/.
	ssh bessel -f screen -S RegressionDlm -dm jdk/bin/java -cp regression_dlm.jar dlm.examples.urbanobservatory.RegressionDlm
	ssh bessel -t screen -ls

temperature_topsy:
	sbt assembly
	cp target/scala-2.11/bayesiandlm-assembly-0.1-SNAPSHOT.jar temperature_dlm.jar
	scp temperature_dlm.jar temperature_model.qsub topsy:/share/nobackup/a9169110/.
	ssh topsy -f "cd /share/nobackup/a9169110 && qsub temperature_model.qsub"
	ssh topsy -t qstat

## Fit a trend+seasonal DLM to the temperature data
temperature_model:
	sbt assembly
	cp target/scala-2.11/bayesiandlm-assembly-0.1-SNAPSHOT.jar temperature_dlm.jar
	scp temperature_dlm.jar bessel:/data/a9169110/.
	ssh bessel -f cd /data/a9169110 && screen -S TemperatureGibbs -dm jdk/bin/java -cp temperature_dlm.jar dlm.examples.urbanobservatory.FitTemperatureModel
	ssh bessel -t screen -ls

## Fit a trend+seasonal DLM to the humidity data
humidity_model:
	sbt assembly
	cp target/scala-2.11/bayesiandlm-assembly-0.1-SNAPSHOT.jar humidity_dlm.jar
	scp humidity_dlm.jar bessel:/data/a9169110/.
	ssh bessel -f cd /data/a9169110 && screen -S TemperatureGibbs -dm jdk/bin/java -cp humidity_dlm.jar dlm.examples.urbanobservatory.HumidityModel
	ssh bessel -t screen -ls

## Fit a joint-trend+seasonal DLM to the humidity and temperature data simultaneously
correlated_seasonal_model:
	sbt assembly
	cp target/scala-2.11/bayesiandlm-assembly-0.1-SNAPSHOT.jar correlated_seasonal_dlm.jar
	scp correlated_seasonal_dlm.jar bessel:/data/a9169110/.
	ssh bessel -f "cd /data/a9169110 && screen -S CorrelatedSeasonalGibbs -dm jdk/bin/java -cp correlated_seasonal_dlm.jar dlm.examples.urbanobservatory.FitJointModel"
	ssh bessel -t screen -ls
