from datetime import datetime, timedelta

import hopsworks
from hsfs.feature_store import FeatureStore
import pandas as pd
import numpy as np

from src import config

def get_hopsworks_project() -> hopsworks.project.Project:

    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

def get_feature_store() -> FeatureStore:

    project = get_hopsworks_project()
    return project.get_feature_store()

def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """"""
    predictions = model.predict(features)

    results = pd.DataFrame()
    results['pickup_location_id'] = features['pickup_location_id'].values
    results['predicted_demand'] = predictions.round(0)

    return results

def load_batch_of_features_from_store(
    current_date: datetime,
) -> pd.DataFrame:
    feature_store = get_feature_store()

    n_features = config.N_FEATURES

    # We want exactly n_features hourly observations
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = fetch_data_to - timedelta(hours=n_features - 1)

    print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION,
    )

    ts_data = feature_view.get_batch_data(
        start_time=fetch_data_from - timedelta(days=1),
        end_time=fetch_data_to + timedelta(days=1),
    )

    # Remove timezone information
    ts_data["pickup_hour"] = (
        pd.to_datetime(ts_data["pickup_hour"])
        .dt.tz_localize(None)
    )

    # Keep only the required time window
    ts_data = ts_data[
        ts_data["pickup_hour"].between(fetch_data_from, fetch_data_to)
    ]

    expected_hours = pd.date_range(fetch_data_from, fetch_data_to, freq="h")

    # Sort by location and time
    ts_data = ts_data.sort_values(
        by=["pickup_location_id", "pickup_hour"]
    )

    location_ids = np.sort(ts_data["pickup_location_id"].unique())

    # Reindex each location so a missing boundary hour is filled with zero.
    # The model was trained on zero-padded hourly slots.
    reindexed_rows = []
    for location_id in location_ids:
        one_location = ts_data.loc[
            ts_data["pickup_location_id"] == location_id,
            ["pickup_hour", "rides"],
        ].set_index("pickup_hour")

        one_location = one_location.reindex(expected_hours, fill_value=0)
        one_location["pickup_location_id"] = location_id
        reindexed_rows.append(
            one_location.reset_index().rename(columns={"index": "pickup_hour"})
        )

    ts_data = pd.concat(reindexed_rows, ignore_index=True)

    # Verify every location has exactly n_features rows
    counts = ts_data.groupby("pickup_location_id").size()

    print("Rows per location:")
    print(counts.describe())

    if not (counts == n_features).all():
        missing = counts[counts != n_features]
        raise ValueError(
            f"Some locations do not have {n_features} rows:\n{missing}"
        )

    # Build feature matrix
    x = np.empty(
        (len(location_ids), n_features),
        dtype=np.float32,
    )

    for i, location_id in enumerate(location_ids):
        rides = (
            ts_data.loc[
                ts_data["pickup_location_id"] == location_id,
                "rides",
            ]
            .to_numpy(dtype=np.float32)
        )

        x[i] = rides

    features = pd.DataFrame(
        x,
        columns=[
            f"rides_previous_{i}_hours"
            for i in range(n_features, 0, -1)
        ],
    )

    features["pickup_hour"] = current_date
    features["pickup_location_id"] = location_ids

    return features


def load_model_from_registry():

    import joblib
    from pathlib import Path

    project = get_hopsworks_project()
    model_registy = project.get_model_registry()

    model = model_registy.get_model(
        name=config.MODEL_NAME,
        version=config.MODEL_VERSION,
    )

    model_dir = model.download()
    model = joblib.load(Path(model_dir) / 'model.pkl')

    return model