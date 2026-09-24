import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(train.iloc[-1], horizon)


def seasonal_naive_forecast(train: pd.Series, horizon: int, season: int = 12) -> np.ndarray:
    last_season = train.iloc[-season:].to_numpy()
    return np.resize(last_season, horizon)


def arima_forecast(train: pd.Series, horizon: int, order=(1, 1, 1)) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(train, order=order).fit()
    return np.asarray(fit.forecast(steps=horizon))


def sarimax_forecast(train: pd.Series, horizon: int, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """Returns (mean forecast, 95% confidence interval DataFrame)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(train, order=order, seasonal_order=seasonal_order).fit(disp=False)
    pred = fit.get_forecast(steps=horizon)
    return np.asarray(pred.predicted_mean), pred.conf_int()


def lstm_forecast(train: pd.Series, horizon: int, window: int = 12, epochs: int = 200,
                  hidden: int = 32, seed: int = 0) -> np.ndarray:
    """Small one-layer LSTM trained on sliding windows, forecasting recursively."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    values = train.to_numpy(dtype=np.float32)
    mean, std = values.mean(), values.std() or 1.0
    scaled = (values - mean) / std

    xs = np.stack([scaled[i:i + window] for i in range(len(scaled) - window)])
    ys = scaled[window:]
    x = torch.tensor(xs).unsqueeze(-1)
    y = torch.tensor(ys).unsqueeze(-1)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, inp):
            out, _ = self.lstm(inp)
            return self.head(out[:, -1])

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        opt.step()

    history = list(scaled[-window:])
    preds = []
    net.eval()
    with torch.no_grad():
        for _ in range(horizon):
            inp = torch.tensor(history[-window:], dtype=torch.float32).view(1, window, 1)
            nxt = net(inp).item()
            preds.append(nxt)
            history.append(nxt)
    return np.array(preds) * std + mean
