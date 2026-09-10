from pathlib import Path

import pandas as pd


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and prepare the selected time-series dataset."""

    df = pd.read_csv(
        data_path,
        parse_dates=["datetime"]
    )

    df = (
        df[["datetime", "value"]]
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return df