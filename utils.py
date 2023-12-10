import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from lockdowndates.core import LockdownDates
from jours_feries_france import JoursFeries
from vacances_scolaires_france import SchoolHolidayDates
import datetime 

problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"
# A type (class) which will be used to create wrapper objects for y_pred


def get_cv(X, y, random_state=0):
    cv = TimeSeriesSplit(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X):
        # Take a random sampling on test_idx so it's that samples are not consecutives.
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)


def get_train_data(path="data/train.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array

def create_x_weather(X):
    weather_wi = pd.read_csv('data/weather_data.csv')
    
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

def generate_lockdown_dates(X):
    date_range = pd.date_range(start='2020-09-01', end='2021-09-09', freq='D')
    df = pd.DataFrame({'Date': date_range})
    lockdown_periods = [('2020-10-31', '2020-12-14'), ('2021-04-04', '2021-05-02')]
    for period_start, period_end in lockdown_periods:
        df.loc[(df['Date'] >= period_start) & (df['Date'] <= period_end), 'Lockdown'] = True

    df['Lockdown'] = df['Lockdown'].fillna(0).map(lambda x: 0 if x == 0 else 1)
    df = df.reset_index().rename(columns={'Date': 'date_merge'})
    X['date_merge'] = pd.to_datetime(X['date'].dt.strftime('%Y-%m-%d'), format='%Y-%m-%d')
    X_ldd = X.merge(df, how = 'left', on = 'date_merge')
    X_ldd = X_ldd.drop(['date_merge'], axis=1)
    
    return X_ldd

def isLockdown(X):
    ld = LockdownDates("France", "2020-09-01", "2021-09-09", ("stay_at_home", "masks"))
    lockdown_dates = ld.dates()
    ld = lockdown_dates.reset_index().rename(columns={'timestamp': 'date_merge'})
    X['date_merge'] = pd.to_datetime(X['date'].dt.strftime('%Y-%m-%d'), format='%Y-%m-%d')
    X_ld = X.merge(ld, how = 'left', on = 'date_merge')
    X_ld = X_ld.drop(['france_masks', 'france_country_code', 'date_merge'], axis=1)
    X_ld['france_stay_at_home'] = X_ld['france_stay_at_home'].map(lambda x: 1 if x == 2 else 0)
    return X_ld.rename(columns = {'france_stay_at_home':'is_lockdown'})

def is_lockdown(X, date_column='date'):
    # Définir les plages de dates à vérifier (sans tenir compte des heures)
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
    columns_to_keep = ['date', 'counter_id','latitude', 'longitude']
    return X[columns_to_keep]


def apply_couvre_feu(df, date_column='date'):

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

def create_rain(X):
    X['rain'] = X['icon_encoded'].map(lambda x: 1 if x == 1 else 0)
    return X

def prepro(X):
    ext_df = pd.read_csv('external_data/external_data.csv')
    ext_df['date'] = pd.to_datetime(ext_df['date'].values.astype('<M8[us]'))
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