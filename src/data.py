from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = ["ticker", "commodity", "date", "open", "high", "low", "close", "volume"]


def load_futures(path: str | Path) -> pd.DataFrame:
    """Load the agricultural futures CSV (one row per commodity per trading day)."""
    df = pd.read_csv(path, parse_dates=["date"])
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df.sort_values(["commodity", "date"]).reset_index(drop=True)


def commodity_series(df: pd.DataFrame, commodity: str, column: str = "close", freq: str = "MS") -> pd.Series:
    """Return one commodity's series resampled to a regular frequency.

    Daily futures data has gaps (weekends, holidays), which leaves the index without a
    frequency; the paper found monthly resampling made ARIMA/SARIMAX far more stable.
    """
    sub = df[df["commodity"] == commodity].set_index("date")[column]
    if sub.empty:
        raise ValueError(f"No rows for commodity {commodity!r}")
    return sub.resample(freq).mean().interpolate(limit_direction="both")


def train_test_split(series: pd.Series, train_frac: float = 0.8) -> tuple[pd.Series, pd.Series]:
    cut = int(len(series) * train_frac)
    return series.iloc[:cut], series.iloc[cut:]
