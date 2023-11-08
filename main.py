#import packages
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
from typing import Dict
import json  

#load model and dictvectorizer
model_file = 'modelrf_n_est_200_maxdepth_5_minleaf_8.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

#Create predict function
def predict(dictvectorizer: DictVectorizer, mlmodel: RandomForestClassifier, datatopredict: Dict):
    x_pred = dictvectorizer.transform([datatopredict])
    y_pred = mlmodel.predict_proba(x_pred)[:,1][0]
    return y_pred


app = FastAPI()

#Define Pydantic Model
class Daydata(BaseModel):
    day: str
    season: str
    weight: float
    bmi: float
    bodyfat_perc: float
    musclemass_perc: float
    lightsleep_sec: float
    deepsleep_sec: float
    remsleep_sec: float
    awake_sec: float
    interruptions: float
    durationtosleep_sec: float
    avg_hr: float
    durationinbed_sec: float
    temp: float



@app.post("/predict/")
async def create_item(daydata: Daydata):
    daydata_dict = dict(daydata)
    y_pred = round(predict(dv,model,daydata_dict),2)
    response = {
        'probability': y_pred,
        'goodsleep': bool((y_pred>=0.5))}
    json_object = json.dumps(response) 
    return json_object


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)