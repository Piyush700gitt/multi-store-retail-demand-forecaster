"""Metrics and chronological split helpers."""
import numpy as np
import pandas as pd


def mae(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def wape(y, p):
    """Weighted absolute percentage error: sum|error| / sum|actual|.
    Stable when individual actuals are small, unlike MAPE."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / np.abs(y).sum())


def metric_row(y, p):
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "WAPE": wape(y, p), "n": int(len(y))}


def split_dates(dates: pd.Series, val_days: int = 42):
    """Return (train_cutoff, val_start). Train uses Date <= cutoff, validation uses Date >= val_start."""
    last = dates.max()
    val_start = last - pd.Timedelta(days=val_days - 1)
    return val_start - pd.Timedelta(days=1), val_start
