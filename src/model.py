import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


SAMPLING_HOURS = 5.0

DOMINANT_PERIOD_HOURS = 274.12

DOMINANT_PERIOD_STEPS = (
    DOMINANT_PERIOD_HOURS / SAMPLING_HOURS
)


FINAL_FEATURE_COLUMNS = [
    "time_normalized",
    "time_squared",
    "sin_cycle",
    "cos_cycle",
    "time_sin_cycle",
    "time_cos_cycle",
]


def create_features(
    df: pd.DataFrame,
    time_scale: float | None = None
) -> tuple[pd.DataFrame, float]:
    """Create the six features used by the final model."""

    features = df[["datetime", "value"]].copy()

    features["time_index"] = np.arange(len(features))

    if time_scale is None:
        time_scale = features["time_index"].max()

    features["time_normalized"] = (
        features["time_index"] / time_scale
    )

    features["time_squared"] = (
        features["time_normalized"] ** 2
    )

    theta = (
        2 * np.pi
        * features["time_index"]
        / DOMINANT_PERIOD_STEPS
    )

    features["sin_cycle"] = np.sin(theta)
    features["cos_cycle"] = np.cos(theta)

    features["time_sin_cycle"] = (
        features["time_normalized"]
        * features["sin_cycle"]
    )

    features["time_cos_cycle"] = (
        features["time_normalized"]
        * features["cos_cycle"]
    )

    return features, time_scale


def train_model(
    feature_data: pd.DataFrame
) -> LinearRegression:
    """Train the final Linear Regression model."""

    model = LinearRegression()

    model.fit(
        feature_data[FINAL_FEATURE_COLUMNS],
        feature_data["value"]
    )

    return model


def create_future_features(
    last_datetime: pd.Timestamp,
    historical_length: int,
    forecast_steps: int,
    time_scale: float
) -> pd.DataFrame:
    """Create features for future forecast timestamps."""

    future_datetime = pd.date_range(
        start=last_datetime + pd.Timedelta(hours=SAMPLING_HOURS),
        periods=forecast_steps,
        freq=f"{SAMPLING_HOURS}h"
    )

    future = pd.DataFrame({
        "datetime": future_datetime
    })

    future["time_index"] = np.arange(
        historical_length,
        historical_length + forecast_steps
    )

    future["time_normalized"] = (
        future["time_index"] / time_scale
    )

    future["time_squared"] = (
        future["time_normalized"] ** 2
    )

    theta = (
        2 * np.pi
        * future["time_index"]
        / DOMINANT_PERIOD_STEPS
    )

    future["sin_cycle"] = np.sin(theta)
    future["cos_cycle"] = np.cos(theta)

    future["time_sin_cycle"] = (
        future["time_normalized"]
        * future["sin_cycle"]
    )

    future["time_cos_cycle"] = (
        future["time_normalized"]
        * future["cos_cycle"]
    )

    return future


def forecast(
    model: LinearRegression,
    future_features: pd.DataFrame
) -> np.ndarray:
    """Generate predictions for future timestamps."""

    return model.predict(
        future_features[FINAL_FEATURE_COLUMNS]
    )