import requests

test_request = {
  "day": "Monday",
  "season": "Autumn",
  "weight": 80,
  "bmi": 25,
  "bodyfat_perc": 24,
  "musclemass_perc": 30,
  "lightsleep_sec": 2000,
  "deepsleep_sec": 1000,
  "remsleep_sec": 500,
  "awake_sec": 500,
  "interruptions": 5,
  "durationtosleep_sec": 3000,
  "avg_hr": 80,
  "durationinbed_sec": 100,
  "temp": 30
}


url = 'http://localhost:8080/predict'


x = requests.post(url, json = test_request)
print(x.content)