from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.monitoring import load_predictions_and_actual_values_from_store

st.set_page_config(layout="wide")

# title
current_date = pd.to_datetime(datetime.utcnow()).floor('H')
st.title(f'Monitoring dashboard 🧭')

progress_bar = st.sidebar.header('🧠 Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 3


@st.cache_data
def _load_predictions_and_actuals_from_store(
    from_date: datetime,
    to_date: datetime
) -> pd.DataFrame:
    """
    Wrapped version of src.monitoring.load_predictions_and_actual_values_from_store, so
    we can add Streamlit caching

    Args:
        from_date (datetime): min datetime for which we want predictions and
            actual values

        to_date (datetime): max datetime for which we want predictions and
            actual values
    """
    return load_predictions_and_actual_values_from_store(from_date, to_date)

with st.spinner(text="Fetching model predictions and actual values from the store"):

    monitoring_df = _load_predictions_and_actuals_from_store(
        from_date=current_date - timedelta(days=14),
        to_date=current_date
    )

    st.sidebar.write('✅ Model predictions and actual values arrived')
    progress_bar.progress(1 / N_STEPS)


with st.spinner(text="Plotting aggregate MAE hour-by-hour"):

    st.header('Mean Absolute Error (MAE) hour-by-hour')

    # MAE per pickup_hour
    monitoring_df['ae'] = (monitoring_df['rides'] - monitoring_df['predicted_demand']).abs()
    mae_per_hour = (
        monitoring_df
        .groupby('pickup_hour', as_index=False)['ae']
        .mean()
        .rename(columns={'ae': 'mae'})
        .sort_values(by='pickup_hour')
    )
    mae_per_hour['pickup_hour'] = mae_per_hour['pickup_hour'].dt.strftime('%Y-%m-%d %H:%M')

    fig = px.bar(
        mae_per_hour,
        x='pickup_hour',
        y='mae',
        template='plotly_dark',
    )

    st.plotly_chart(
        fig,
        theme="streamlit",
        width='stretch',
    )

    progress_bar.progress(2 / N_STEPS)


with st.spinner(text="Plotting MAE hour-by-hour for top locations"):

    st.header('Mean Absolute Error (MAE) per location and hour')

    top_locations_by_demand = (
        monitoring_df
        .groupby(['pickup_location_id'])['rides']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .head(10)['pickup_location_id']
    )

    for location_id in top_locations_by_demand:

        mae_per_hour = (
            monitoring_df[monitoring_df.pickup_location_id == location_id]
            .groupby('pickup_hour', as_index=False)['ae']
            .mean()
            .rename(columns={'ae': 'mae'})
            .sort_values(by='pickup_hour')
        )
        mae_per_hour['pickup_hour'] = mae_per_hour['pickup_hour'].dt.strftime('%Y-%m-%d %H:%M')

        fig = px.bar(
            mae_per_hour,
            x='pickup_hour',
            y='mae',
            template='plotly_dark',
        )

        st.subheader(f'{location_id=}')
        st.plotly_chart(
            fig,
            theme="streamlit",
            width='stretch',
        )

    progress_bar.progress(3 / N_STEPS)