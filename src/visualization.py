import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast(
    historical: pd.DataFrame,
    future: pd.DataFrame,
    output_path
) -> None:
    """Plot historical signal and future forecast."""

    plt.figure(figsize=(15, 6))

    plt.plot(
        historical["datetime"],
        historical["value"],
        label="Historical"
    )

    plt.plot(
        future["datetime"],
        future["forecast"],
        label="6-Month Forecast"
    )

    plt.axvline(
        historical["datetime"].iloc[-1],
        linestyle="--",
        label="Forecast Start"
    )

    plt.xlabel("Datetime")
    plt.ylabel("Value")
    plt.title("Periodic Signal — Six-Month Forecast")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()