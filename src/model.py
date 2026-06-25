import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline


import lightgbm as lgb


def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    X['average_rides_last_4_weeks'] = 0.25 * (
        X[f'rides_previous_{7 * 24}_hours']
        + X[f'rides_previous_{2 * 7 * 24}_hours']
        + X[f'rides_previous_{3 * 7 * 24}_hours']
        + X[f'rides_previous_{4 * 7 * 24}_hours']
    )

    # Same hour yesterday and last week
    X['rides_same_hour_yesterday'] = X[f'rides_previous_24_hours']
    X['rides_same_hour_1_week_ago'] = X[f'rides_previous_{7 * 24}_hours']
    X['rides_same_hour_2_weeks_ago'] = X[f'rides_previous_{2 * 7 * 24}_hours']

    # Short-term trend: mean of last 3h vs mean of hours 4–6
    X['short_term_trend'] = (
        X['rides_previous_1_hours'] + X['rides_previous_2_hours'] + X['rides_previous_3_hours']
    ) / 3 - (
        X['rides_previous_4_hours'] + X['rides_previous_5_hours'] + X['rides_previous_6_hours']
    ) / 3

    # 24h rolling mean (last 24 hours)
    lag_24_cols = [f'rides_previous_{i}_hours' for i in range(1, 25)]
    X['mean_rides_last_24h'] = X[lag_24_cols].mean(axis=1)

    return X


class TemporalFeaturesEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X_ = X.copy()
        X_['pickup_hour'] = pd.to_datetime(X_['pickup_hour'], utc=True)

        X_['hour'] = X_['pickup_hour'].dt.hour
        X_['day_of_week'] = X_['pickup_hour'].dt.dayofweek
        X_['month'] = X_['pickup_hour'].dt.month
        X_['day_of_month'] = X_['pickup_hour'].dt.day
        X_['is_weekend'] = (X_['pickup_hour'].dt.dayofweek >= 5).astype(int)
        X_['week_of_year'] = X_['pickup_hour'].dt.isocalendar().week.astype(int)

        return X_.drop(columns=['pickup_hour'])


def get_pipeline(**hyperparams) -> Pipeline:

    add_feature_average_rides_last_4_weeks = FunctionTransformer(
        func=average_rides_last_4_weeks,
        validate=False
    )

    add_temporal_features = TemporalFeaturesEngineer()

    return make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyperparams)
    )