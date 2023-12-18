import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import utils

# from pathlib import Path
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

# PREPROCESSING AND MERGE WEATHER DATA
X, y = utils.get_train_data()

#
num_features = ["temp", "precip", "windspeed", "visibility"]
cat_features = ["counter_id"]
time_features = ["hour", "month", "weekday", "day"]

col_transformer = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(sparse=False), cat_features),
        ("time", FunctionTransformer(utils.encode_cyclical_features), time_features),
    ],
    remainder="passthrough",
)

# Temporal Train Test Split
X_train, y_train, X_test, y_test = utils.train_test_split_temporal(
    X, y, delta_threshold="30 days"
)

# Create the Model
cat = CatBoostRegressor(depth=12, iterations=1500, rsm=0.35, subsample=0.7, verbose=0)

# Create pipeline with preprocessing (merge data), column_transformer, and CatBoostRegressor
pipe = Pipeline(
    [
        ("prepro", FunctionTransformer(utils.prepro)),
        ("col", col_transformer),
        ("model", cat),
    ]
)

# Fitting the model
pipe.fit(X_train, y_train)

predictions = pipe.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error: {rmse}")


# Refitting on the whole dataset for Kaggle Submission
pipe.fit(X, y)
y_pred = pipe.predict(X_test)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)

# Exporting results
results.to_csv("new_submission.csv", index=False)
