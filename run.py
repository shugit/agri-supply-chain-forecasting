"""End-to-end run: load data -> resample -> fit models -> evaluate -> save metrics and plots."""
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import commodity_series, load_futures, train_test_split
from src.evaluate import metrics
from src.features import adf_test, stl_decompose
from src.models import (arima_forecast, lstm_forecast, naive_forecast,
                        sarimax_forecast, seasonal_naive_forecast)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=os.environ.get("AGRI_DATA", "data/all_agricultural_products_data.csv"))
    p.add_argument("--commodities", nargs="+", default=["Coffee", "Cocoa", "Sugar", "Random Length Lumber"])
    p.add_argument("--out", default="outputs")
    p.add_argument("--skip-lstm", action="store_true")
    return p.parse_args()


def run_one(df, commodity, out: Path, skip_lstm: bool):
    series = commodity_series(df, commodity)
    train, test = train_test_split(series)
    h = len(test)

    forecasts = {
        "naive": naive_forecast(train, h),
        "seasonal_naive": seasonal_naive_forecast(train, h),
        "ARIMA(1,1,1)": arima_forecast(train, h),
    }
    sarimax_mean, sarimax_ci = sarimax_forecast(train, h)
    forecasts["SARIMAX(1,1,1)(1,1,1,12)"] = sarimax_mean
    if not skip_lstm:
        forecasts["LSTM"] = lstm_forecast(train, h)

    rows = [{"commodity": commodity, "model": name, **metrics(test, pred)} for name, pred in forecasts.items()]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train.index, train, label="train", color="black")
    ax.plot(test.index, test, label="actual", color="gray")
    for name, pred in forecasts.items():
        ax.plot(test.index, pred, label=name, linestyle="--")
    ax.fill_between(test.index, sarimax_ci.iloc[:, 0], sarimax_ci.iloc[:, 1], alpha=0.15, label="SARIMAX 95% CI")
    ax.set_title(f"{commodity}: monthly close price, 80/20 train/test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"forecast_{commodity.replace(' ', '_')}.png", dpi=120)
    plt.close(fig)

    stl_decompose(series).plot().savefig(out / f"stl_{commodity.replace(' ', '_')}.png", dpi=120)
    plt.close("all")

    return rows, {"commodity": commodity, **adf_test(series), **{f"seasonal_diff_{k}": v for k, v in adf_test(series.diff(12)).items()}}


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_futures(args.data)

    all_rows, adf_rows = [], []
    for commodity in args.commodities:
        rows, adf = run_one(df, commodity, out, args.skip_lstm)
        all_rows += rows
        adf_rows.append(adf)

    results = pd.DataFrame(all_rows)
    results.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(adf_rows).to_csv(out / "adf_tests.csv", index=False)
    print(results.round(2).to_string(index=False))
    (out / "metrics.json").write_text(json.dumps(all_rows, indent=2))


if __name__ == "__main__":
    main()
