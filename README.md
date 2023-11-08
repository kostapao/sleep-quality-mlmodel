# sleep-quality-ml-model

At home I use a sleep tracker from Withings which every day yields a certain sleepscore. The sensors track heart rate, sleep duration, time in the sleephases, interruptions etc.
I enriched that data with weather data (temperature during the night), weight data from my Garmin Smart Scale and seasonality data. The goal was to create an ML Model with which theoratically one could predict good sleep vs bad sleep.

* sleepscore_analysis.ipynb contains data exploration and modelling
* main.py contains the fast api app
* to use the model navigate to the location of the docker file and then run
  1. docker build -t sleep-model .
  2. docker run -p 8080:8080 sleep-model

* To test the model run the predict.py script
