import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(
    actual,
    predicted
) -> dict:
    """Calculate MAE and RMSE."""

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
    }