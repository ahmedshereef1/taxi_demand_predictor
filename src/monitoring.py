from datetime import datetime, timedelta

import pandas as pd

import src.config as config
from src.feature_store_api import get_feature_store, get_feature_group


def load_predictions_and_actual_values_from_store(
    from_date: datetime,
    to_date: datetime,
) -> pd.DataFrame:
    """
    Fetches model predictions and actuals values from
    `from_date` to `to_date` from the Feature Store and returns a dataframe

    Args:
        from_date (datetime): min datetime for which we want predictions and
            actual values

        to_date (datetime): max datetime for which we want predictions and
            actual values

    Returns:
        pd.DataFrame: 4 columns
            - `pickup_location_id`
            - `predicted_demand`
            - `pickup_hour`
            - `rides`
    """
    predictions_fg = get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTIONS)
    actuals_fg = get_feature_group(name=config.FEATURE_GROUP_NAME)

    feature_store = get_feature_store()

    # fetch predictions via its own feature view
    predictions_fv_name = config.FEATURE_GROUP_MODEL_PREDICTIONS + '_fv'
    try:
        feature_store.create_feature_view(
            name=predictions_fv_name,
            version=1,
            query=predictions_fg.select_all(),
        )
    except:
        pass

    predictions_fv = feature_store.get_feature_view(name=predictions_fv_name, version=1)
    predictions_df = predictions_fv.get_batch_data(
        start_time=from_date - timedelta(days=1),
        end_time=to_date + timedelta(days=1),
    )

    # fetch actuals via its own feature view
    actuals_fv_name = config.FEATURE_GROUP_NAME + '_fv'
    try:
        feature_store.create_feature_view(
            name=actuals_fv_name,
            version=1,
            query=actuals_fg.select_all(),
        )
    except:
        pass

    actuals_fv = feature_store.get_feature_view(name=actuals_fv_name, version=1)
    actuals_df = actuals_fv.get_batch_data(
        start_time=from_date - timedelta(days=1),
        end_time=to_date + timedelta(days=1),
    )

    # strip timezone from both
    for df in [predictions_df, actuals_df]:
        if df['pickup_hour'].dt.tz is not None:
            df['pickup_hour'] = df['pickup_hour'].dt.tz_convert(None)

    # left join: keep all predictions, attach actuals where available
    monitoring_df = predictions_df.merge(
        actuals_df[['pickup_hour', 'pickup_location_id', 'rides']],
        on=['pickup_hour', 'pickup_location_id'],
        how='left',
    )

    print(f'Predictions: {len(predictions_df)} rows | Actuals joined: {monitoring_df["rides"].notna().sum()} non-null')

    return monitoring_df[['pickup_location_id', 'predicted_demand', 'pickup_hour', 'rides']]
