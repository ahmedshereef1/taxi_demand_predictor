import zipfile
from datetime import datetime

import requests
import pandas as pd
import numpy as np

import streamlit as st
import geopandas as gpd
import pydeck as pdk

from src.inference import (
    load_batch_of_features_from_store,
    load_model_from_registry,
    get_model_predictions
)

from src.paths import DATA_DIR
from src.plot import plot_one_example

st.set_page_config(layout='wide')

# title
# current_date = datetime.striptime('2026-01-05 12:00:00')
current_data = pd.to_datetime(datetime.utcnow()).floor('h')
st.title(f'Taxi demand prediction')
st.header(f'{current_data}')

progress_bar = st.sidebar.header('Working Progress')
progress_bar =st.sidebar.progress(0)
N_STEPS = 7 

def load_shape_data_file():

    URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

    response = requests.get(URL)

    path = DATA_DIR / "taxi_zones.zip"

    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
    else:
        raise Exception(f"{URL} is not available")

    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR / "taxi_zones")

    return gpd.read_file(
        DATA_DIR / "taxi_zones" / "taxi_zones" / "taxi_zones.shp"
    ).to_crs("EPSG:4326")

with st.spinner(text='Downloading shape file to plot taxi zones'):
    geo_df = load_shape_data_file()
    st.sidebar.write('Shape file was downloaded')
    progress_bar.progress(1/N_STEPS)

with st.spinner(text="Fetching batch of inference data"):
    features = load_batch_of_features_from_store(current_data)
    st.sidebar.write('Inference features fetched from the store')
    progress_bar.progress(2/N_STEPS)
    print(f'{features}')

with st.spinner(text="Loading ML model from the registry"):
    model = load_model_from_registry()

    print(model)

    if hasattr(model, "feature_names_in_"):
        print("Number of expected features:", len(model.feature_names_in_))
        print(model.feature_names_in_)

    st.sidebar.write("ML model was downloaded from registry")
    progress_bar.progress(3 / N_STEPS)

with st.spinner(text="Computing model predictions"):

    print("Inference shape:", features.shape)
    print(features.columns.tolist())

    results = get_model_predictions(model, features)

    st.sidebar.write("Model prediction arrived")
    progress_bar.progress(4 / N_STEPS)

with st.spinner(text="Preparing data to plot"):

    def demand_to_color(val, minval, maxval):
        """Map demand value to a blue → yellow → red heatmap color."""
        if maxval == minval:
            return [0, 200, 0, 160]

        t = (val - minval) / (maxval - minval)

        r = 0
        g = int(80 + t * 175)
        b = 0
        alpha = int(120 + t * 135)  

        return [r, g, b, alpha]

    df = pd.merge(
        geo_df,
        results,
        left_on="LocationID",
        right_on="pickup_location_id"
    )

    max_pred = df["predicted_demand"].max()
    min_pred = df["predicted_demand"].min()

    df["fill_color"] = df["predicted_demand"].apply(
        lambda x: demand_to_color(x, min_pred, max_pred)
    )

    # scale max height to 800m for dramatic 3D effect
    df["elevation"] = (
        (df["predicted_demand"] - min_pred) / (max_pred - min_pred + 1e-9)
    ) * 800

    progress_bar.progress(5 / N_STEPS)


with st.spinner(text="Generating NYC Map"):

    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=40.7549,
        longitude=-73.9840,
        zoom=10.5,
        max_zoom=16,
        pitch=50,
        bearing=-10,
    )

    geojson = pdk.Layer(
        "GeoJsonLayer",
        data=df,
        opacity=0.9,
        stroked=True,
        filled=True,
        extruded=True,
        wireframe=False,
        get_elevation="elevation",
        elevation_scale=1,
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255, 30],
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 60],
    )

    tooltip = {
        "html": """
        <div style="
            font-family: -apple-system, sans-serif;
            font-size: 13px;
            padding: 4px 2px;
        ">
            <div style="font-size:15px; font-weight:600; margin-bottom:6px;">
                {zone}
            </div>
            <div style="color:#94a3b8; margin-bottom:2px;">Borough</div>
            <div style="margin-bottom:8px;">{borough}</div>
            <div style="color:#94a3b8; margin-bottom:2px;">Predicted demand</div>
            <div style="font-size:20px; font-weight:700; color:#fbbf24;">
                {predicted_demand}
            </div>
        </div>
        """,
        "style": {
            "backgroundColor": "#1e293b",
            "color": "#f1f5f9",
            "borderRadius": "8px",
            "padding": "12px 16px",
            "boxShadow": "0 4px 20px rgba(0,0,0,0.4)",
        },
    }

    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        initial_view_state=INITIAL_VIEW_STATE,
        layers=[geojson],
        tooltip=tooltip,
    )

    st.pydeck_chart(deck, use_container_width=True)

    st.sidebar.write("NYC map generated")
    progress_bar.progress(6 / N_STEPS)

with st.spinner(text="Plotting time-series data"):

    row_indices = np.argsort(results["predicted_demand"].values)[::-1]
    n_to_plot = 10

    # Plot each time series with its prediction
    for row_id in row_indices[:n_to_plot]:
        fig = plot_one_example(
            features=features,
            target=results["predicted_demand"],
            example_id=row_id,
            predictions=results["predicted_demand"],
        )

        st.plotly_chart(
            fig,
            theme="streamlit",
            width="stretch",
            config={"displayModeBar": False},
        )

    progress_bar.progress(7 / N_STEPS)
    st.sidebar.success("Done!")