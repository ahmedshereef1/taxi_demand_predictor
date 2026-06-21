import pandas as pd

from datetime import datetime
from typing import Tuple

def train_test_split(
    df: pd.DataFrame,
    cutoff_data: datetime,
    target_colum_name: str,
) ->Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splitting the dataset into train and test split based on a datetime cutoff.

    Args:
        df (pd.DataFrame): The input DataFrame containing the full dataset,
                           must include a 'pickup_hour' column of datetime type.
        cutoff_data (datetime): The datetime threshold used to split the data.
                                Rows before this value go to train, rows from
                                this value onward go to test.
        target_colum_name (str): The name of the target column to be predicted.
                                 This column is separated from the features.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
            - X_train (pd.DataFrame): Training features (rows where pickup_hour < cutoff_data).
            - y_train (pd.Series): Training target values.
            - X_test (pd.DataFrame): Test features (rows where pickup_hour >= cutoff_data).
            - y_test (pd.Series): Test target values.
    """

    train_data = df[df.pickup_hour < cutoff_data].reset_index(drop=True)
    test_data = df[df.pickup_hour >= cutoff_data].reset_index(drop=True)

    X_train = train_data.drop(columns=target_colum_name)
    y_train = train_data[target_colum_name]

    X_test = test_data.drop(columns=target_colum_name)
    y_test = test_data[target_colum_name]

    return X_train, y_train, X_test, y_test