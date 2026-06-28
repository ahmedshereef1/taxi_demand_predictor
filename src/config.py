import os
from dotenv import load_dotenv
from src.paths import PARENT_DIR

# loads keyvalue pairs from .env
load_dotenv(PARENT_DIR / ".env")

HOPSWORKS_PROJECT_NAME= "taxi_demand_prediction"
try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception("Create an .env file on the project root with the HOPSWORKS_API_KEY")

FEATURE_GROUP_NAME = "time_series_hourly_feature_group"
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = "time_series_hourly_feature_view"
FEATURE_VIEW_VERSION = 1


# 29 days × 24 hours
N_FEATURES  = 29 * 24  # 696
MODEL_NAME = "taxi_demand_predictor_next_hour"
MODEL_VERSION = 2

FEATURE_GROUP_MODEL_PREDICTIONS = "model_predictions"

FEATURE_VIEW_MONITORING = "monitoring_feature_view"

MAX_MAE = 15.0