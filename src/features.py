import warnings

import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller


def add_lags(series: pd.Series, lags=(1, 2)) -> pd.DataFrame:
    frame = series.to_frame("y")
    for lag in lags:
        frame[f"lag{lag}"] = series.shift(lag)
    return frame.dropna()


def adf_test(series: pd.Series) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        stat, pvalue, used_lag, nobs, *_ = adfuller(series.dropna())
    return {"adf_statistic": stat, "p_value": pvalue, "lags_used": used_lag, "n_obs": nobs}


def stl_decompose(series: pd.Series, period: int = 12):
    """Trend / seasonal / residual split; period=12 for monthly data with yearly seasonality."""
    return STL(series, period=period, robust=True).fit()
