from datetime import datetime
from loguru import logger

import pandas as pd

from src.data import load_row_data, tranfrom_row_data_into_ts_data
from src.config import FEATURE_GROUP_NAME
from src.feature_store_api import get_feature_group


def get_historical_rides() -> pd.DataFrame:
    """
    Download historical rides from NYC taxi dataset
    """
    from_year = 2025
    to_year = datetime.now().year
    print(f'Downloading raw data from {from_year} to {to_year}')

    rides = pd.DataFrame()
    for year in range(from_year, to_year+1):

        # download data for the whole year
        rides_one_year = load_row_data(year)
        
        # append rows
        rides = pd.concat([rides, rides_one_year])

    return rides


def run():

    logger.info('Fetching raw data from data warehouse')    
    rides = get_historical_rides()

    logger.info('Transforming raw data into time-series data')
    ts_data = tranfrom_row_data_into_ts_data(rides)

    ts_data['pickup_hour'] = pd.to_datetime(ts_data['pickup_hour'], utc=True)

    # get a pointer to the feature group we wanna write to
    feature_group = get_feature_group(FEATURE_GROUP_NAME)

    # start a job to insert the data into the feature group
    logger.info('Inserting data into feature group...')
    feature_group.insert(
        ts_data,
        write_options={"wait_for_job": True}
    )

if __name__ == '__main__':
    run()