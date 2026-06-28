import requests
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime, timedelta

from tqdm import tqdm
import pandas as pd
import numpy as np

from src.paths import RAW_DATA_DIR


def fetch_batch_raw_data(from_date: datetime, to_date: datetime) -> pd.DataFrame:
    """
    Simulate production data by sampling historical data from 52 weeks ago.
    """
    from_date_ = from_date - timedelta(days=7*52)
    to_date_ = to_date - timedelta(days=7*52)

    rides = load_row_data(year=from_date_.year, months=from_date_.month)
    rides = rides[rides.pickup_datetime >= from_date_]
    rides_2 = load_row_data(year=to_date_.year, months=to_date_.month)
    rides_2 = rides_2[rides_2.pickup_datetime < to_date_]

    rides = pd.concat([rides, rides_2])

    # shift data forward 52 weeks to simulate recent data
    rides['pickup_datetime'] += timedelta(days=7*52)

    rides.sort_values(by=['pickup_location_id', 'pickup_datetime'], inplace=True)

    return rides

def dowanlod_row_file_of_raw_data(year: int, month: int) -> Path:
    """
    Download a monthly NYC taxi trip dataset and save it locally.

    Args:
        year (int):
            Dataset year.
        month (int):
            Dataset month.

    Returns:
        Path:
            Path to the downloaded parquet file.
    """
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"

    response = requests.get(url)

    if response.status_code == 200:
        path = RAW_DATA_DIR / f"rides_{year}-{month:02d}.parquet"
        open(path, "wb").write(response.content)
        return path
    else:
        raise Exception(f"{url} is not available. Status code: {response.status_code}")
    

def validate_raw_data(
    rides: pd.DataFrame,
    year: int,
    month: int,
) -> pd.DataFrame:
    """
    Validate and filter taxi ride data for a specific month.

    This function removes any records whose pickup_datetime falls
    outside the specified year and month.

    Args:
        rides (pd.DataFrame):
            DataFrame containing taxi ride records. Must include a
            'pickup_datetime' column.
        year (int):
            Target year for filtering.
        month (int):
            Target month for filtering.

    Returns:
        pd.DataFrame:
            A filtered DataFrame containing only rides that occurred
            within the specified month.
    """
    # keep only rides for this month
    this_month_start = f'{year}-{month:02d}-01'
    
    if month < 12:
        next_month_start = f"{year}-{month + 1:02d}-01"
    else:
        next_month_start = f"{year + 1}-01-01"

    rides = rides[rides.pickup_datetime >= this_month_start]
    rides = rides[rides.pickup_datetime < next_month_start]

    return rides


def load_row_data(
    year: int,
    months: Optional[List[int]] = None
) -> pd.DataFrame:
    
    rides = pd.DataFrame()

    if months is None:
        # Dowanlod data only for the months specified by 'months'
        months = list(range(1,13))
    elif isinstance(months,int):
        months = [months]
    
    for month in months:

        local_file = RAW_DATA_DIR / f"rides_{year}-{month:02d}.parquet"
        if not local_file.exists():
            try:
                # Dowanlod the data from NYC Website
                print(f"Downloading file {year}-{month:02d}")
                dowanlod_row_file_of_raw_data(year, month)
            except:
                print(f'{year}-{month:02d} is not available')
                continue  
        else:
            print(f'file {year}-{month:02d} was already in local storage')
        
        # load the file into Pandas
        rides_one_month = pd.read_parquet(local_file)

        # rename columns
        rides_one_month = rides_one_month[['tpep_pickup_datetime','PULocationID']]
        rides_one_month.rename(columns={
                "tpep_pickup_datetime": "pickup_datetime",
                "PULocationID": "pickup_location_id"
            },
            inplace=True
        )

        # validate the file
        rides_one_month = validate_raw_data(rides_one_month, year=year, month=month)

        # append to existing data
        rides = pd.concat([rides, rides_one_month])

    # keep only time and origin of the ride
    rides = rides[['pickup_datetime','pickup_location_id']]

    return rides

def add_missing_solts(rides: pd.DataFrame) -> pd.DataFrame:

    locations_ids = rides['pickup_location_id'].unique()
    full_range = pd.date_range(
        rides['pickup_hour'].min(), rides['pickup_hour'].max(), freq='h'
    )

    output = pd.DataFrame()

    for locations_id in tqdm(locations_ids):

        # keep only rides for this 'location_id'
        rides_i = rides.loc[rides.pickup_location_id == locations_id, ['pickup_hour','rides'] ]

        # Adds missing dates with 0 in a Series
        rides_i.set_index('pickup_hour', inplace=True)
        rides_i.index = pd.DatetimeIndex(rides_i.index)
        rides_i = rides_i.reindex(full_range, fill_value=0)

        # add back location_id colums
        rides_i['pickup_location_id'] = locations_id

        output = pd.concat([output, rides_i])

    output = output.reset_index().rename(columns={'index': 'pickup_hour'})

    return output


def tranfrom_row_data_into_ts_data(
    rides: pd.DataFrame
) -> pd.DataFrame:
    """"""
    # sum rides per loaction and pickup_datetime
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('h')

    agg_rides = rides.groupby(['pickup_location_id', 'pickup_hour']).size().reset_index()
    agg_rides.rename(columns={0: 'rides'}, inplace=True)

    agg_rides_all_sslots = add_missing_solts(agg_rides)

    return agg_rides_all_sslots


def get_cutoff_indices(
    data: pd.DataFrame,
    n_features: int,
    step_size: int
) -> list:
    """
    Generates the indices needed to create sliding windows for a time-series dataset.
    """
    stop_position = len(data) - 1

    # Start from first sub_sequence at index position 0
    subseq_first_idx = 0
    subseq_mid_index = n_features
    subseq_last_idx = n_features + 1

    indices = []
    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_mid_index, subseq_last_idx))

        subseq_first_idx += step_size
        subseq_mid_index += step_size
        subseq_last_idx += step_size
    
    return indices

def transform_ts_data_info_feature_and_target(
    ts_data: pd.DataFrame,
    input_seq_len: int,
    step_size: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Slice and Transform data from time-series format into a (feature,target)
    foramt that we use it to train Supervised ML models.
    """
    assert set(ts_data.columns) == {'pickup_hour','rides', 'pickup_location_id'}
    
    location_ids = ts_data['pickup_location_id'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()

    for location_id in tqdm(location_ids):

        # keep only ts_data for 'location_id'
        ts_data_one_location = ts_data.loc[
            ts_data.pickup_location_id == location_id,['pickup_hour','rides']
        ]

        # Compute cutoff indices to split dataframe rows
        indices = get_cutoff_indices(
            ts_data_one_location,
            input_seq_len,
            step_size,
        )

         # Slice and transpose data into numpy array for features and target
        n_samples = len(indices)
        x = np.ndarray(shape=(n_samples, input_seq_len), dtype=np.float32)
        y = np.ndarray(shape=(n_samples), dtype=np.float32) 

        pickup_hours = []

        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['rides'].values[0]
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

        # numpy -> pandas
        feature_one_location = pd.DataFrame(
            x,
            columns=[f'rides_previous_{i+1}_hours' for i in reversed(range(input_seq_len))]
        )
        feature_one_location['pickup_hour'] = pickup_hours
        feature_one_location['pickup_location_id'] = location_id

        # numpy -> pandas
        target_one_location = pd.DataFrame(
            y, columns=[f'target_rides_next_hour']
        )

        # Concatenate results 
        features = pd.concat([features, feature_one_location])
        targets = pd.concat([targets, target_one_location])
    
    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)
    
    return features, targets['target_rides_next_hour'] 
