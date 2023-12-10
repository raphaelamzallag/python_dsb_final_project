# %% [code] {"execution":{"iopub.status.busy":"2023-12-10T20:48:26.790960Z","iopub.execute_input":"2023-12-10T20:48:26.791480Z","iopub.status.idle":"2023-12-10T20:48:26.800610Z","shell.execute_reply.started":"2023-12-10T20:48:26.791439Z","shell.execute_reply":"2023-12-10T20:48:26.799196Z"}}
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
#import utils
#from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from jours_feries_france import JoursFeries
from vacances_scolaires_france import SchoolHolidayDates
from catboost import CatBoostRegressor
import datetime
from datetime import timedelta, date

# **PREPROCESSING AND MERGE WEATHER DATA**

#Normally all these functions are in the utils.py file

_target_column_name = "log_bike_count"

def get_train_data(path="/kaggle/input/mdsb-2023/train.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array

def create_x_weather(X):
    """
    Merge weather data with X
    """
    weather_wi = pd.read_csv('/kaggle/input/weather-data/weather_data.csv')

    columns_to_keep = ['datetime', 'temp', 'precip', 'windspeed', 'visibility']
    weather = weather_wi[columns_to_keep].copy().rename(columns={'datetime':'date'})

    mapping = {'snow': 0, 'rain': 1, 'cloudy': 2, 'partly-cloudy-day': 3, 'clear-day': 4}
    weather.loc[:, 'icon_encoded'] = weather_wi['icon'].copy().map(mapping)


    weather['date'] = pd.to_datetime(weather['date'].values.astype('<M8[us]'), format='%Y-%m-%d')
    weather['date_merge'] = weather['date']
    X_weather = X.copy()
    X_weather['date_merge'] = pd.to_datetime(X_weather['date'].dt.strftime('%Y-%m-%d'), format='%Y-%m-%d')
    X_weather = X_weather.merge(weather.drop(columns=['date']), how='left', on='date_merge').drop(columns=['date_merge'])

    return X_weather

def is_jour_ferie(X):
    """
    Add binary columns is_jour_ferie and is_holidays
    """
    d = SchoolHolidayDates()
    ho_20 = d.holidays_for_year_and_zone(2020, 'A')
    ho_21 = d.holidays_for_year_and_zone(2021, 'A')

    jf_20 = JoursFeries.for_year(2020)
    jf_21 = JoursFeries.for_year(2021)

    df_jf_21 =pd.DataFrame(jf_21.keys(),jf_21.values(),columns=['fete']).reset_index().rename(columns={'index':'date'})
    df_jf_20 =pd.DataFrame(jf_20.keys(),jf_20.values(),columns=['fete']).reset_index().rename(columns={'index':'date'})
    df_jf = pd.concat([df_jf_20,df_jf_21]).reset_index().drop('index',axis=1)
    df_jf['date'] = pd.to_datetime(df_jf['date'], format='%Y-%m-%d')

    df_ho20 = pd.DataFrame({'date': key, 'vacances': value['vacances_zone_a']} for key, value in ho_20.items())
    df_ho21 = pd.DataFrame({'date': key, 'vacances': value['vacances_zone_a']} for key, value in ho_20.items())
    df_ho = pd.concat([df_ho20,df_ho21]).reset_index().drop('index',axis=1)
    df_ho['date'] = pd.to_datetime(df_ho['date'], format='%Y-%m-%d')


    X['date_merge'] = pd.to_datetime(X['date'].dt.strftime('%Y-%m-%d'), format='%Y-%m-%d')

    X_final = X.merge(df_jf, how='left', left_on='date_merge', right_on='date', suffixes=('','_drop')).drop(columns={'date_drop'})
    X_final['is_ferie'] = X_final['fete'].map(lambda x: 0 if pd.isna(x) else 1)

    X_final = X_final.merge(df_ho, how='left', left_on='date_merge', right_on='date', suffixes=('','_drop')).drop(columns={'date_drop'})
    X_final['is_vacances'] = X_final['vacances'].map(lambda x: 0 if pd.isna(x) else 1)

    return X_final.drop_duplicates().drop(['date_merge','fete','vacances'],axis=1)

def is_lockdown(X, date_column='date'):
    """
    Function to create binary column is_lockdown
    """
    date_ranges = [
        {"start": datetime.datetime(2020, 3, 17), "end": datetime.datetime(2020, 5, 11)},
        {"start": datetime.datetime(2020, 10, 30), "end": datetime.datetime(2020, 12, 15)},
        {"start": datetime.datetime(2021, 4, 3), "end": datetime.datetime(2021, 5, 3)}
    ]

    # Fonction pour vérifier si une date est dans l'un des intervalles
    def is_date_in_range(date):
        if not isinstance(date, str):
            date = date.strftime('%Y-%m-%d')

        dt = datetime.datetime.strptime(date, '%Y-%m-%d')
        for period in date_ranges:
            if period["start"].date() <= dt.date() <= period["end"].date():
                return 1
        return 0

    # Appliquer la fonction à la colonne 'date' et créer une nouvelle colonne pour le résultat
    X['is_lockdown'] = X[date_column].apply(is_date_in_range)

    return X



def encode_cyclical_features(df):
    """
    function to encode cyclical features
    """
    columns = ['hour', 'day', 'month', 'weekday']
    max_value = {'hour': 24, 'day': 31, 'month': 31, 'weekday': 7}

    for column in df:
        if column in columns:
            df[column + '_sin'] = np.sin(2 * np.pi * df[column]/max_value[column])
            df[column + '_cos'] = np.cos(2 * np.pi * df[column]/max_value[column])
            df.drop(columns=[column], inplace=True)
    return df

def train_test_split_temporal(X, y, delta_threshold="30 days"):
    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = X["date"] <= cutoff_date
    X_train, X_test = X.loc[mask], X.loc[~mask]
    y_train, y_test = y[mask], y[~mask]

    return X_train, y_train, X_test, y_test

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour
    X["is_weekend"] = X["weekday"].map(lambda x: 1 if x == 6 or x == 5 else 0)
    # Finally we can drop the original columns from the dataframe
    return X#.drop(columns=["date"])

def filter_columns(X):
    """
    Filter columns at the beginning of preprocessing
    """
    columns_to_keep = ['date', 'counter_id','latitude', 'longitude']
    return X[columns_to_keep]


def apply_couvre_feu(df, date_column='date'):
    """
    Create binary column is_couvre_feu
    """
    couvre_feu_periods = [
        {"start": datetime.datetime(2020, 10, 17), "end": datetime.datetime(2020, 10, 29), "start_time": datetime.time(21, 0), "end_time": datetime.time(6, 0)},
        {"start": datetime.datetime(2021, 1, 16), "end": datetime.datetime(2021, 3, 20), "start_time": datetime.time(18, 0), "end_time": datetime.time(6, 0)},
            {"start": datetime.datetime(2021, 3, 21), "end": datetime.datetime(2021, 5, 19), "start_time": datetime.time(19, 0), "end_time": datetime.time(6, 0)},
            {"start": datetime.datetime(2021, 5, 20), "end": datetime.datetime(2021, 6, 9), "start_time": datetime.time(21, 0), "end_time": datetime.time(6, 0)},
            {"start": datetime.datetime(2021, 6, 10), "end": datetime.datetime(2021, 6, 20), "start_time": datetime.time(23, 0), "end_time": datetime.time(6, 0)}
    ]
    def is_couvre_feu(date_heure):

        if not isinstance(date_heure, str):
            date_heure = date_heure.strftime('%Y-%m-%d %H:%M')

        dt = datetime.datetime.strptime(date_heure, '%Y-%m-%d %H:%M')
        for period in couvre_feu_periods:
            if period["start"].date() <= dt.date() <= period["end"].date():
                if ((dt.time() >= period["start_time"]) or (dt.time() < period["end_time"])):
                    return 1
        return 0

    df['is_couvre_feu'] = df[date_column].apply(is_couvre_feu)
    return df

def prepro(X):
    """
    Function aggregating all the preprocessing functions
    """
    data = is_jour_ferie(
                is_lockdown(
                    create_x_weather(
                        apply_couvre_feu(
                            filter_columns(X)
                            )
                        )
                    )
                )
    result = _encode_dates(data).drop(columns=['date'])
    return result

# %% [code] {"execution":{"iopub.status.busy":"2023-12-10T20:48:28.201519Z","iopub.execute_input":"2023-12-10T20:48:28.201985Z","iopub.status.idle":"2023-12-10T20:48:28.514597Z","shell.execute_reply.started":"2023-12-10T20:48:28.201938Z","shell.execute_reply":"2023-12-10T20:48:28.513268Z"}}
X, y = get_train_data()

# %% [code] {"execution":{"iopub.status.busy":"2023-12-10T20:48:28.776455Z","iopub.execute_input":"2023-12-10T20:48:28.776936Z","iopub.status.idle":"2023-12-10T20:48:28.820638Z","shell.execute_reply.started":"2023-12-10T20:48:28.776893Z","shell.execute_reply":"2023-12-10T20:48:28.819363Z"}}
num_features = ['temp', 'precip', 'windspeed', 'visibility']
cat_features = ['counter_id']
time_features = ['hour','month','weekday','day']

col_transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(sparse=False), cat_features),
        ('time', FunctionTransformer(encode_cyclical_features), time_features)
    ],
    remainder='passthrough'
)

X_train, y_train, X_test, y_test = train_test_split_temporal(X, y, delta_threshold="30 days")

# %% [code] {"execution":{"iopub.status.busy":"2023-12-10T20:48:29.757563Z","iopub.execute_input":"2023-12-10T20:48:29.758020Z"}}
cat = CatBoostRegressor(
    depth=12,
    iterations=1500,
    rsm=0.35,
    subsample=0.7,
    verbose=0
)

pipe = Pipeline([
    ('prepro',FunctionTransformer(prepro)),
    ('col', col_transformer),
    ('model', cat)
])


pipe.fit(X_train, y_train)

predictions = pipe.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'Root Mean Squared Error: {rmse}')

# %% [code]
pipe.fit(X, y)
y_pred = pipe.predict(X_test)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)
