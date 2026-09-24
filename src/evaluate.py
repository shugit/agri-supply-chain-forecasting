import numpy as np


def metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = actual - predicted
    mse = float(np.mean(err ** 2))
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE_%": float(np.mean(np.abs(err / actual)) * 100),
    }
