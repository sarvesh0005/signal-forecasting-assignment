from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_data
from src.model import (
    SAMPLING_HOURS,
    create_features,
    create_future_features,
    forecast,
    train_model,
)
from src.visualization import plot_forecast


def main():

    project_root = Path(__file__).resolve().parent

    data_path = (
        project_root
        / "data"
        / "raw"
        / "timeseries_5h.csv"
    )

    plot_path = (
        project_root
        / "plots"
        / "final_forecast.png"
    )

    # Load data
    df = load_data(data_path)

    # Create features
    feature_data, time_scale = create_features(df)

    # Train final model
    model = train_model(feature_data)

    # Calculate six-calendar-month horizon
    forecast_end = (
        df["datetime"].max()
        + pd.DateOffset(months=6)
    )

    forecast_steps = int(
        np.ceil(
            (
                forecast_end
                - df["datetime"].max()
            ).total_seconds()
            / (SAMPLING_HOURS * 3600)
        )
    )

    # Create future features
    future_data = create_future_features(
        last_datetime=df["datetime"].iloc[-1],
        historical_length=len(df),
        forecast_steps=forecast_steps,
        time_scale=time_scale,
    )

    # Generate forecast
    future_data["forecast"] = forecast(
        model,
        future_data
    )

    # Save visualization
    plot_forecast(
        historical=df,
        future=future_data,
        output_path=plot_path,
    )

    print("Forecast completed.")
    print(f"Historical observations: {len(df)}")
    print(f"Forecast steps: {forecast_steps}")
    print(f"Forecast end: {future_data['datetime'].iloc[-1]}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()