# Agricultural Supply Chain Forecasting

Forecasting agricultural commodity prices from historical futures data, to support stocking and supply decisions. The project follows the methods of the paper *Enhancing Supply Chain Efficiency with Time Series Analysis and Deep Learning Techniques* (Sun, Zhou, Zhan, and Wu, 2024).

This repository is a clean re-implementation of the methods described in the paper by its authors, written for public release; results may differ slightly from those reported.

## What it does

For each commodity (by default coffee, cocoa, sugar, and random-length lumber), the pipeline:

1. Loads daily futures data and resamples the closing price to monthly averages. Daily futures data has gaps on weekends and holidays; the paper found monthly resampling made the statistical models much more stable.
2. Tests stationarity with the augmented Dickey–Fuller test, before and after a 12-month seasonal difference.
3. Splits each series 80/20 into training and test periods.
4. Fits five forecasters and scores them on the test period:
   - naive (last value) and seasonal naive (same month last year) baselines,
   - ARIMA(1,1,1),
   - SARIMAX(1,1,1)(1,1,1,12), with a 95% confidence interval,
   - a small one-layer LSTM trained on 12-month sliding windows.
5. Saves metrics (MSE, RMSE, MAE, MAPE), forecast plots, and STL trend/seasonal/residual decompositions to `outputs/`.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data   # put all_agricultural_products_data.csv here (see "Data")
python run.py                                  # all default commodities
python run.py --commodities Coffee --skip-lstm # faster, statistical models only
python run.py --data /path/to/file.csv --out results/
```

`notebooks/forecasting_walkthrough.ipynb` walks through the same steps for one commodity.

## Data

The paper uses the agricultural futures market dataset from Kaggle: daily trading data from January 3, 2000 to December 11, 2023 for cocoa, coffee, cotton, orange juice, random-length lumber, and sugar futures. The expected file is `all_agricultural_products_data.csv` with columns:

`ticker, commodity, date, open, high, low, close, volume`

The dataset is available from Kaggle under its own license and is kept out of this repository. Download it from Kaggle (search "agricultural futures market data 2000–2023") and place it in `data/`.

## Results from this re-implementation

Monthly closing price, 80/20 split, one run on the dataset above (lower is better):

| Commodity | Model | RMSE | MAPE % |
|---|---|---|---|
| Coffee | Naive | 75.66 | 32.39 |
| Coffee | ARIMA(1,1,1) | 72.41 | 30.63 |
| Coffee | SARIMAX | 70.50 | 29.70 |
| Coffee | LSTM | 60.02 | 24.70 |
| Sugar | Naive | 7.44 | 30.69 |
| Sugar | SARIMAX | 6.81 | 27.52 |
| Sugar | LSTM | 6.05 | 24.49 |
| Cocoa | Naive | 1904.24 | 14.54 |
| Cocoa | SARIMAX | 1745.43 | 14.70 |
| Cocoa | LSTM | 1782.46 | 18.64 |
| Random Length Lumber | Seasonal naive | 312.14 | 35.55 |
| Random Length Lumber | SARIMAX | 374.28 | 30.59 |
| Random Length Lumber | LSTM | 424.35 | 38.87 |

For coffee and sugar, SARIMAX and the LSTM improve on the baselines. Cocoa's test period includes the 2023 price surge, which every model misses to a similar degree. For lumber, the seasonal baseline has the lowest RMSE.

The paper reports ARIMA mean squared errors of 53,738.74 on the raw daily series and 1,926.81 after monthly resampling and feature engineering. Its exact test commodity and settings for those figures are not fully specified, so the numbers above are not directly comparable.

## Status / known limitations

- The model settings follow the paper's description; they are not tuned per commodity.
- The LSTM is intentionally small and uses a fixed seed; results vary somewhat with seed and epochs.
- The lasso and random-forest feature-importance analysis described in the paper is not included yet.

## Project layout

```
run.py                 end-to-end pipeline
src/data.py            loading, monthly resampling, train/test split
src/features.py        lag features, ADF test, STL decomposition
src/models.py          naive, seasonal naive, ARIMA, SARIMAX, LSTM
src/evaluate.py        MSE, RMSE, MAE, MAPE
notebooks/             walkthrough notebook
```

## Citation

```bibtex
@article{sun2024enhancing,
  title   = {Enhancing Supply Chain Efficiency with Time Series Analysis and Deep Learning Techniques},
  author  = {Sun, Jun and Zhou, Shuwen and Zhan, Xiaoan and Wu, Jiang},
  journal = {Preprints},
  year    = {2024},
  doi     = {10.20944/preprints202409.0983.v1}
}
```

## License

MIT (see `LICENSE`). The dataset is subject to its own license on Kaggle.
