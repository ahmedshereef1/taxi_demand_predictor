## Taxi Demand Predictor Service

- My interest in creating this project was ignited after reading UBER's blog post on (:link: [Demand and ETR Forecasting at Airports](https://www.uber.com/en-GB/blog/demand-and-etr-forecasting-at-airports/))

## Table of Contents
  * [Quick Setup](#quick-setup)
  * [Problem Statement](#problem-statement)
  * [Data Processing](#data-processing)
  * [Model Training](#model-training)
  * [MLOps](#mlops)


## Quick Setup

1. Install [Python Poetry](https://python-poetry.org/)
    ```
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2. cd into the project folder and run
    ```
    $ poetry install
    ```

3. Activate the virtual env that you just created with
    ```
    $ poetry shell
    ```

4. Open free accounts at Hopsworks and CometML and copy your project names and API keys in an .env file
    ```
    $ cp .env.sample .env
    # paste your 2 values there
    ```

5. Backfill the feature group with historical data
    ```
    $ make backfill
    ```

6. Run the training pipeline
    ```
    $ make training
    ```

7. Run the feature pipeline for the last hour
    ```
    $ make features
    ```

8. Run the inference pipeline to generate predictions for the last hour
    ```
    $ make predictions
    ```

## Problem Statement

- You work as a data scientist in a ride-sharing app company (e.g. Uber)

- Your job is to help the operations team **keep the fleet as busy as possible**.
    - (You want drivers to be working as much as possible)
    - Every driver that is not working is an opportunity that you're losing.
    - So essentially, the company's end goal is to keep current fleet as productive as it can be

### Supply and Demand Problem

- Consider Central Park in Manhattan, every day, at every moment, there are two things that you need to balance
    - The total number of drivers that are around Central Park at that time, hour by hour.
    - The blue line shows the number of drivers who are available to pick up users around the Central Park
    - The orange lines indicates the users who are looking for a ride
    - What you want in order to maximize the number of pickups, in order to keep the fleet as busy as possible, is to match these two curves
<p align="left">
<img src="images/supply_demand_chart1.png"/>
</p>

- So we need to balance both excess of drivers and lack of drivers
- So as a data scientist, your job is to make these curves as close as possible.
- You can only control the no. of drivers but not no. of users requesting for rides. But what you do is predict. This is where ML enters into the picture
- Imagine you'll be developing an ML model that is going to predict how many users are going to look for a taxi ride in different areas of New York at each point in time.
- If you have this model you could plan ahead and rearrange the distribution of the fleet in order to match the demand
<p align="left">
<img src="images/supply_demand_chart2.png"/>
</p>

- How are we gonna build the ML model?
- We'll use historical data of taxi rides, or more precisely, people looking for taxi ride
- But we don't have this data instead we have data of actual rides.
- The question is, can we use that info to predict what's going to happen in the next hour.
- So, at each point in time we'll have a set of historical features that we can use to generate our estimate(prediction)
- What to predict? - What is going to happen - no. of rides that our users will request in each area of New York city in next hour
<p align="left">
<img src="images/supply_demand_chart3.png"/>
</p>


## Data Preparation

**Step 1 - Data Validation**

- Let's see the steps that take us from raw data to nice formatted data that we need to do ML
- We're gonna start with raw data that we fetch from external website (historical taxi rides, saying certain taxi ride happened in that part of NY at that time).
- We collect such events
- First thing after collecting the raw data is to validate it. You have to do this in every data pipeline i.e. make sure that the events that you're using are correct
- Features: pickup_datetime, pickup_location_id
<p align="left">
<img src="images/data_validation.png"/>
</p>


**Step 2 - Raw data into time-series data**
- Post cleaned data, we need to transform it into time-series data (aggregate a list of raw events into time series data)
- We bucket events per hour and per area in NYC
- This way we get collection of time series data
- create new feature called `rides` = Count no. of rides per pickup_hour per pickup_location_id
- To make sure you have complete timeseries, add missing rows for which rides didn't happen for that pickup_hour and pickup_location_id
<p align="left">
<img src="images/timeseries_transform.png"/>
</p>


**Step 3 - Time-series data into (features, target) data**
- To apply ML, we need features and target
- We transform the TS data through slicing operations into the right format to perform supervised ML
- We take sliding window of 24 (24 hours) and use it to predict the next hour
<p align="left">
<img src="images/sliding_window.png"/>
</p>

**Step 4 - From raw data to training data**
- Put the above 3 steps(notebooks) together to construct the entire `data pipeline` that is going to ingest the raw data and outputs training data that is features and targets
<p align="left">
<img src="images/data_pipeline.png"/>
</p>

**Step 5 - Explore and visualize the final dataset**

<p align="left">
<img src="images/dataset_exploration.png"/>
</p>



## Model Training
- Split the data based on the cutoff date (since it's a time series data) into training and test data
- Create a baseline model: A simple rule that you infer just by looking at the data that uses no ML, has no complexities to obtain baseline performance
- We use MAE since it's a regression problem
- We test this baseline model on the test data (This is baseline performance)
- After baseline model, develop ML model. If the MAE of ML model is smaller to baseline model then we have built a better model. This is an iterative process by building multiple versions of the model

<p align="left">
<img src="images/model_building_strategy.png"/>
</p>

Q) **How to build this sequence of models? (Strategies to find better models)**
Here are 4 ways to improve your model:
1) **Increase training data size**
- If we trained 1st model on 2 yrs data, we can train new model with 1 yr of data

<p align="left">
<img src="images/improve_training_data.png"/>
</p>

2) **Add more features to the training data**
- Try to use external factors like calendar holidays in US (Patterns during holidays in taxi demand changes a lot. `Eg:` Spike in taxi demand during Christmas)
- Adding useful features can help the model get better results
<p align="left">
<img src="images/improve_features.png"/>
</p>

3) **Try powerful algorithms**
- Boosting algorithm works best with tabular dataset
<p align="left">
<img src="images/improve_algorithms.png"/>
</p>

4) **Hyperparameter tuning**

<p align="left">
<img src="images/improve_hyperparams.png"/>
</p>

**Note:**
- Sometimes people spend way too much time on hyperparameter tuning. Instead they can add more features (holiday features, weather features etc) to the model or perform feature engineering, i.e. from initial raw features you derive new features from them.
- For `eg:` You could add a feature that computes the trend in average demand in the last two weeks. So, instead of giving hourly values of taxi demand for that area, you could also provide an idea of the trend. If we have upward trend, then this maybe a valuable signal for the model
So, engineering features is always something that works better or has larger impact than tuning hyperparameters


    <p align="left">
<img src="images/feature_engineering_avg.png"/>
</p>

- Derive numerical features from `pickup_hour`

<p align="left">
<img src="images/feature_engineering_hour.png"/>
</p>

- This is how it'll look after deriving new features `hour`, `week_day` from `pickup_hour`

<p align="left">
<img src="images/feature_engineering_derived.png"/>
</p>

- `Proposed idea:` Use pickup_location_id as a categorical encoding.

<p align="left">
<img src="images/feature_engineering_location.png"/>
</p>

- These are areas representing location in NYC so instead of using these numbers we create `latitude` and `longitude` of some middle point in that area and use them as features. This way we provide a representation of the data that respects distances on map

<p align="left">
<img src="images/feature_engineering_latlon.png"/>
</p>

### Scikit-learn Pipeline
- Useful to package different steps
- Steps to create the pipeline
    - Define the scikit-learn pipeline
    - Train the model pipeline
    - Predict

<p align="left">
<img src="images/sklearn_pipeline.png"/>
</p>

### Hyperparameter Tuning with Optuna
- Create N splits of the training data. Here we create 4 splits. It's a tradeoff between the number of splits and the time it takes to train the model
- Start with 1st split to train and 2nd split to validate to get MAE1
- Then, take 1st two splits to train and 3rd split to validate to get MAE2
- Finally, take 1st three splits to train and 4th split to validate to get MAE3
- Average all 3 MAEs. Averaging errors is a better estimation of the actual error
- Once, you're happy with hyperparameters, you retrain the model using entire training data and at the end you test it

<p align="left">
<img src="images/hyperparameter_tuning.png"/>
</p>

### MLOps

### Model Deployment as Batch-scoring System

- It is a sequence of steps of computing and storage that map recent data to predictions that can be used by the business
- Batch scoring system has the following pipelines
    - Data Preparation pipeline or Feature pipeline
    - Model Training pipeline
    - Prediction pipeline

**Pipeline 1 - Data Preparation pipeline or Feature pipeline**

- This component runs every hour
- For eg: every hour, we extract raw data from an external service - from a data warehouse or wherever the recent data is
- Once we fetch raw data, we then create a tabular dataset with features and target and store them in the feature store
- Feature Store is a key component in this system that is a part of any modern MLOps platform
- This is the Data Ingestion Pipeline

<p align="left">
<img src="images/feature_pipeline.png"/>
</p>

**Step 2 - Train ML Model**
- 2nd pipeline - `Model Training pipeline`
- Retrain the model since ML models in real-world systems are trained regularly
- In this project, it's on-demand, whenever I think I want to train the model, I can trigger this pipeline, and it automatically trains, generates a new model and saves it back to the model registry

<p align="left">
<img src="images/training_pipeline.png"/>
</p>

**Step 3 - Generate predictions on recent data**
- 3rd pipeline - `Prediction pipeline`
- Use most recent features and current model we have in production to generate predictions
- Use these predictions to display on UI using streamlit application and deploy
<p align="left">
<img src="images/prediction_pipeline.png"/>
</p>


**Serverless MLOps tools**

- **`Hopsworks`** as our feature store
   - It's a serverless platform that provides an infrastructure to manage and run the feature store automatically
   - It's easy to manage unlike GCP, Azure where we have to setup different components first

- **`Github Actions`** to schedule and run jobs
   - We automate the feature pipeline that will ingest data every hour
   - The notebook is going to automatically run every hour and it's going to fetch a batch of recent data, transform it and save it into features store
   - Created a configuration yaml file under `.github/workflows`
   - The cron job runs every hour
   - The command below triggers the notebook execution from command line


**Feature Store**
- Feature store is used to store features.
- These features can be used to either train the models or make predictions.
- Features saved in the feature store are:
   - `pickup_hour`
   - `no_of_rides`
   - `pickup_location_id`

<p align="left">
<img src="images/feature_store.png"/>
</p>


**Backfill the Feature Store**
- Fetch files from the year 2025
- Transform raw data into time series data
- Dump it in the feature store
- Repeat for the year 2026 and so on
