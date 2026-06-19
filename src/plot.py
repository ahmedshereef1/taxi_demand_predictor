import pandas as pd
import plotly.express as px

from typing import Optional

def plot_one_example(
    features: pd.DataFrame,
    target: pd.Series,
    example_id: int,
    predictions: Optional[pd.Series] = None
): 
    """
    Plot one example of the training data.
    """
    features_ = features.iloc[example_id]
    target_ = target.iloc[example_id]

    ts_columns = [col for col in features.columns if col.startswith('rides_previous_')]
    ts_values = [ features_[c]  for c in ts_columns] + [target_]
    ts_dates = pd.date_range(
        start=features_['pickup_hour'] - pd.Timedelta(hours=len(ts_columns)),
        end=features_['pickup_hour'],
        freq='h',
    )

    # line plot with last values
    title = f'Pick up hour={features_["pickup_hour"]}, location_id={features_["pickup_location_id"]}'
    fig = px.line(
        x=ts_dates,
        y=ts_values,
        title=title,
        markers=True,
    )
    
    # green dot for the value we wanna predict
    fig.add_scatter(
        x=ts_dates[-1:],
        y=[target_],
        line_color='green',
        mode='markers',
        marker_color='green',
        marker_size=10,
        name='actual value',
    )

    if predictions is not None:
        # big red X for the predicted values, if passed
        predictions_ = predictions.iloc[example_id]
        fig.add_scatter(
            x=ts_dates[-1:],
            y=[predictions_],
            line_color='red',
            mode='markers',
            name='predicted value',
            marker_size=15,
        )
    
    return fig