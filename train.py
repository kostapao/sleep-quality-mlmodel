
# # Sleep Score Withings Data
# 
# At home I use a sleep tracker from withing which every day yields a certain sleepscore. The sensors track heart rate, sleep duration, time in the sleephases, interruptions etc.
# I will enrich that data with weather data (temperature during the night), weight data from my Garmin Smart Scale and seasonality data. The goal is to create an ML Model with which theoratically could predict good sleep vs bad sleep. A sleep score above 85 will be seen as good sleep.
# 
# 
# * Create Dataset
# * EDA

#Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import pickle

##### Data Processing
# ## Create Dataset

#Read Data
df_weight = pd.read_csv('weight_garmin.csv')
df_sleep_stats = pd.read_csv('sleep_stats_withings.csv')
df_weather = pd.read_csv('weather.csv')
#This will be the variable to predict
df_sleepscore = pd.read_excel('sleepscore.xlsx')
df_seasonality = pd.DataFrame(columns = ['date','day','season'])

#Find relevant date range
start_date = df_sleepscore['date'].min()
end_date = df_sleepscore['date'].max()

#Create numpy array with daterange start to end

daterange = np.arange(start_date, end_date, np.timedelta64(1,'D'))

#Create empty dataframe with dates to join
df_date = pd.DataFrame()
df_date['date'] = daterange



### Process Weight

#Weight dataframe shows date with nan values and then the actual values below with time measured

#Get the index of all dates, apply regex finding values with a year in it '\d{4}'
df_weight_dates = df_weight[df_weight['Zeit'].str.contains('\d{4}', regex = True)].Zeit.values
df_weight_dates_indices = df_weight[df_weight['Zeit'].str.contains('\d{4}', regex = True)].index


#Get the indices with values with date index + 1
df_weight_measurements_indices = list(df_weight_dates_indices + 1)
df_weight_measurements = df_weight.iloc[df_weight_measurements_indices,:]
df_weight_measurements = df_weight_measurements.rename(columns = {'Zeit':'date','Gewicht':'weight', 'BMI':'bmi', 'Körperfett':'bodyfat_perc','Skelettmuskelmasse':'musclemass_perc'}).copy()
df_weight_measurements.loc[:,'date'] = df_weight_dates

df_weight_measurements.reset_index(inplace = True)

df_weight_measurements = df_weight_measurements.loc[:,['date','weight','bmi','bodyfat_perc','musclemass_perc']]

#Process Date column, make date format

df_weight_measurements['date'] =  df_weight_measurements['date'].apply(lambda x: x.strip())

#Month dictionary
months_DE_EN = {'Okt': 'Oct', 'Mai':'May','Mrz':'Mar', 'Dez':'Dec'}

translated_dates = []
for val in df_weight_measurements['date'].values:
    month_translated = False
    for month in months_DE_EN:
        if month in val:
            val = val.replace(month,months_DE_EN[month])
            translated_dates.append(val)
            month_translated = True
    if month_translated == True:
        pass
    else:
        translated_dates.append(val)

df_weight_measurements['date'] = translated_dates
df_weight_measurements['date'] = pd.to_datetime(df_weight_measurements['date'],format='%b %d, %Y')

#Process Other columns

df_weight_measurements['weight'] = df_weight_measurements['weight'].apply(lambda x: x.replace(' kg',''))
df_weight_measurements['weight'] = pd.to_numeric(df_weight_measurements['weight'], errors='coerce')

df_weight_measurements['bmi'] = pd.to_numeric(df_weight_measurements['bmi'], errors='coerce')

df_weight_measurements['bodyfat_perc'] = df_weight_measurements['bodyfat_perc'].apply(lambda x: x.replace(' %',''))
df_weight_measurements['bodyfat_perc'] = pd.to_numeric(df_weight_measurements['bodyfat_perc'], errors='coerce')

df_weight_measurements['musclemass_perc'] = df_weight_measurements['musclemass_perc'].apply(lambda x: x.replace(' kg',''))
df_weight_measurements['musclemass_perc'] = pd.to_numeric(df_weight_measurements['musclemass_perc'], errors='coerce')

#Check if there are any NA
#df_weight_measurements.isna().sum()

#Fill NA with previous values
df_weight_measurements['bodyfat_perc'] = df_weight_measurements['bodyfat_perc'].ffill()
df_weight_measurements['musclemass_perc'] = df_weight_measurements['musclemass_perc'].ffill()


#Check if there are any NA and types are correct
#df_weight_measurements.info()
df_weight = df_weight_measurements

# #### Process Weather Data

#Select street nearest to me and lets only take temperature
df_weather = df_weather[(df_weather['Standort']=='Zch_Schimmelstrasse') & (df_weather['Einheit']=='°C')]
#Only Take relevant columns and rename
df_weather = df_weather.loc[:,['Datum','Wert']]
df_weather = df_weather.rename(columns={'Datum':'date','Wert':'temp'})

#The relevant temperature for sleep should only be the temperature in the nighttime, we will take the average temperature from midnight till 7am, for this we need to create an hour column, then remove the hours we are not interested in, group by date and take the average
df_weather['date'] = pd.to_datetime(df_weather['date'])
df_weather['hour']= df_weather.loc[:,'date'].dt.hour
df_weather['date'] = df_weather.loc[:,'date'].dt.date

#Remove all hours that are not between 0 and 8 (inclusive)
df_weather = df_weather[(df_weather['hour']>=0)&(df_weather['hour']<=8)]


#Group and take average
df_weather = df_weather.loc[:,df_weather.columns != 'hour']
df_weather = df_weather.groupby('date').mean()

df_weather.reset_index(drop=False, inplace = True)
df_weather.temp = df_weather.temp.round(2)
df_weather


#Check for nas
df_weather.isna().sum()


#Fill nas with previous value
df_weather['temp'] = df_weather['temp'].ffill()

df_weather['date'] = pd.to_datetime(df_weather['date'])


# #### Process Sleep Stats

df_sleep_stats = df_sleep_stats.loc[:, ['bis', 'leicht (s)', 'tief (s)', 'rem (s)', 'wach (s)',
       'Aufwachen', 'Duration to sleep (s)', 'Average heart rate']]

df_sleep_stats.rename(columns= {'bis':'date', 'leicht (s)': 'lightsleep_sec', 'tief (s)' : 'deepsleep_sec', 'rem (s)':'remsleep_sec', 'wach (s)':'awake_sec',
       'Aufwachen':'interruptions', 'Duration to sleep (s)':'durationtosleep_sec', 'Average heart rate':'avg_hr'}, inplace = True)

df_sleep_stats['durationinbed_sec'] = df_sleep_stats['deepsleep_sec'] + df_sleep_stats['remsleep_sec'] + df_sleep_stats['lightsleep_sec'] + df_sleep_stats['awake_sec']


df_sleep_stats['date'] = [date[:10] for date in df_sleep_stats['date'].values]
df_sleep_stats['date'] = pd.to_datetime(df_sleep_stats['date'])

#Check if there are nas and check types
#df_sleep_stats.info()


# #### Process Seasonality Data

df_seasonality['date'] = daterange
df_seasonality['day'] = df_seasonality['date'].dt.day_name()
df_seasonality['month'] = df_seasonality['date'].dt.month
season = []
for month in df_seasonality['month'].values:
    if month in [3,4,5]:
        season.append('spring')
    elif month in [2,1,12]:
        season.append('winter')
    elif month in [6,7,8]:
        season.append('summer')
    else:
        season.append('autumn')

df_seasonality['season'] = season

df_seasonality = df_seasonality.loc[:,df_seasonality.columns!='month']

# #### Process Sleepscore

#df_sleepscore.info()

#46 days without sleepscore
df_sleepscore[df_sleepscore['sleepscore'].isna()]
# For simplicity reasons we will put the mean when data is missing
mean = df_sleepscore[df_sleepscore['sleepscore'].isna() == False].sleepscore.mean()
df_sleepscore= df_sleepscore.fillna(mean)

# #### Join all dataframes

df = df_seasonality.merge(df_weight, how='left').merge(df_sleep_stats, how = 'left').merge(df_weather, how = 'left').merge(df_sleepscore, how = 'left')

#df.isna().sum()

#For all the weight measruments missing we will take previous value

df['weight'] = df['weight'].ffill()
df['bmi'] =  df['bmi'].ffill()
df['bodyfat_perc'] = df['bodyfat_perc'].ffill()
df['musclemass_perc'] = df['musclemass_perc'].ffill()

#for all the sleep tracking values missing we will take mean

df['lightsleep_sec'] = df['lightsleep_sec'].fillna(df['lightsleep_sec'].mean())
df['deepsleep_sec'] = df['deepsleep_sec'].fillna(df['deepsleep_sec'].mean())
df['remsleep_sec'] = df['remsleep_sec'].fillna(df['remsleep_sec'].mean())
df['awake_sec'] = df['awake_sec'].fillna(df['awake_sec'].mean())
df['interruptions'] = df['interruptions'].fillna(df['interruptions'].mean())
df['durationtosleep_sec'] = df['durationtosleep_sec'].fillna(df['durationtosleep_sec'].mean())
df['durationinbed_sec'] = df['durationinbed_sec'].fillna(df['durationinbed_sec'].mean())
df['avg_hr'] = df['avg_hr'].fillna(df['avg_hr'].mean())

df['sleepquality'] = (df['sleepscore'] >= 85).astype(int)

df = df.loc[:,df.columns!='sleepscore']
df = df.loc[:,df.columns!='date']


##### Model Data Prep

df_full_train_orig, df_test_orig = train_test_split(df, train_size=0.8, test_size=0.2, random_state=1)
df_train_orig, df_val_orig = train_test_split(df_full_train_orig, train_size=0.75, test_size=0.25, random_state=1)


#Working on copies of the original dataframes
df_full_train = df_full_train_orig.copy()
df_train = df_train_orig.copy()
df_val = df_val_orig.copy()
df_test = df_test_orig.copy()


y_full_train = df_full_train.sleepquality
y_train = df_train.sleepquality
y_val = df_val.sleepquality
y_test = df_test.sleepquality

#Remove y from train and test dataframes

del df_full_train['sleepquality']
del df_train['sleepquality']
del df_val['sleepquality']
del df_test['sleepquality']


#####One Hot Encoding

#Instantiate dict vectorizer
dv = DictVectorizer(sparse=False)

#Transform df to list(dict)

df_full_train_dict = df_full_train.to_dict(orient='records')
df_train_dict = df_train.to_dict(orient='records')
df_val_dict = df_val.to_dict(orient='records')
df_test_dict = df_test.to_dict(orient='records')


X_fulltrain = dv.fit_transform(df_full_train_dict)
X_train = dv.fit_transform(df_train_dict)
X_val = dv.transform(df_val_dict)
X_test = dv.transform(df_test_dict)

#####Create Model

n_est = 200
max_depth = 5
min_leaf = 8

rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, min_samples_leaf=min_leaf,random_state=1)

rf.fit(X_fulltrain,y_full_train)
#Predict
y_pred = rf.predict_proba(X_test)[:,1]
#Evaluate
auc = roc_auc_score(y_test, y_pred)

print(f'The final random forest model applied on the test data has auc: {round(auc,2)}')


output_file = f'modelrf_n_est_{n_est}_maxdepth_{max_depth}_minleaf_{min_leaf}.bin'

#Save the model and the DictVectorizer
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)

#

