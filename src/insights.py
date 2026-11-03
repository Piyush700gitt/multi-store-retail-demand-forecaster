"""UI-independent helpers for the dashboard: forecast calendar, alert rules, summaries.
No Streamlit imports here, so everything is testable."""
import numpy as np
import pandas as pd

from . import forecasting as FC

HIGH_DEFAULT = 1.20
LOW_DEFAULT = 0.80
REF_WINDOW_DAYS = 56
MIN_REF_OBS = 3
RECENT_DAYS = 28
PROMO_PERIOD = 14
PROMO_MODES = ("none", "repeat_14d")


def last_history_date(history):
    return pd.Timestamp(history["Date"].max()).normalize()


def allowed_start_range(history):
    """Forecasts must start after the data ends (no in-sample 'forecasts')."""
    last = last_history_date(history)
    return last + pd.Timedelta(days=1), last + pd.Timedelta(days=1 + FC.MAX_GAP_DAYS)


def _repeat_promo(history, store_id, dates, period=PROMO_PERIOD):
    """Assumption: the store's Promo flag repeats every `period` days from its last observed pattern."""
    by_date = history[history["Store"] == store_id].set_index("Date")["Promo"]
    last = last_history_date(history)
    out = []
    for t in pd.DatetimeIndex(dates):
        k = int(np.ceil((t - last).days / period))
        out.append(int(by_date.get(t - pd.Timedelta(days=period * k), 0)))
    return np.array(out)


def build_calendar(history, store_id, start, horizon, promo_mode="none"):
    """Default known-in-advance calendar for one store, from the day after the last
    history date through the end of the horizon (the span forecast_stores requires)."""
    if promo_mode not in PROMO_MODES:
        raise ValueError(f"promo_mode must be one of {PROMO_MODES}")
    start = pd.Timestamp(start).normalize()
    past = history[history["Date"] < start]
    if past.empty:
        raise FC.ForecastError("No history before the forecast start date.")
    first = past["Date"].max() + pd.Timedelta(days=1)
    dates = pd.date_range(first, start + pd.Timedelta(days=int(horizon) - 1))
    cal = FC.default_future_calendar(past, [int(store_id)], dates)
    cal = cal.sort_values("Date").reset_index(drop=True)
    if promo_mode == "repeat_14d":
        cal["Promo"] = _repeat_promo(history, int(store_id), cal["Date"])
    return cal


def reference_levels(history, store_id, start):
    """Median sales per weekday (1=Mon..7=Sun) over the last 56 days before the forecast."""
    start = pd.Timestamp(start).normalize()
    last_hist = history.loc[history["Date"] < start, "Date"].max()
    if pd.isna(last_hist):
        return pd.Series(dtype=float)
    h = history[(history["Store"] == store_id) & (history["Date"] < start)
                & (history["Date"] > last_hist - pd.Timedelta(days=REF_WINDOW_DAYS))
                & (history["is_trainable"] == 1)]
    g = h.groupby(h["Date"].dt.dayofweek + 1)["Sales"].agg(["median", "size"])
    return g.loc[g["size"] >= MIN_REF_OBS, "median"]


def recent_level(history, store_id, start):
    """Mean actual sales per open day over the last 28 days before the forecast."""
    start = pd.Timestamp(start).normalize()
    last_hist = history.loc[history["Date"] < start, "Date"].max()
    if pd.isna(last_hist):
        return np.nan
    h = history[(history["Store"] == store_id) & (history["Date"] < start)
                & (history["Date"] > last_hist - pd.Timedelta(days=RECENT_DAYS))
                & (history["is_trainable"] == 1)]
    return float(h["Sales"].mean()) if len(h) else np.nan


def flag_days(fc, ref, high=HIGH_DEFAULT, low=LOW_DEFAULT):
    """Transparent rule: ratio = predicted sales / store's usual sales for that weekday.
    High demand if ratio >= high, Low demand if ratio <= low. Closed days are never flagged."""
    if not (0 < low < 1 < high):
        raise ValueError("Need 0 < low < 1 < high.")
    out = fc.copy()
    out["Reference"] = out["DayOfWeek"].map(ref)
    out["Ratio"] = out["PredictedSales"] / out["Reference"]
    is_open = out["Open"] == 1
    out["Alert"] = "Normal"
    out.loc[is_open & out["Ratio"].isna(), "Alert"] = "No reference"
    out.loc[is_open & (out["Ratio"] >= high), "Alert"] = "High demand"
    out.loc[is_open & (out["Ratio"] <= low), "Alert"] = "Low demand"
    out.loc[~is_open, "Alert"] = "Closed"
    out.loc[~is_open, "Ratio"] = np.nan
    return out


def summarize(flagged):
    op = flagged[flagged["Open"] == 1]
    out = {"total": float(flagged["PredictedSales"].sum()), "n_days": int(len(flagged)),
           "n_open": int(len(op)), "n_closed": int((flagged["Open"] == 0).sum()),
           "avg_open": np.nan, "peak_date": None, "peak_value": np.nan,
           "trough_date": None, "trough_value": np.nan,
           "n_high": int((flagged["Alert"] == "High demand").sum()),
           "n_low": int((flagged["Alert"] == "Low demand").sum()),
           "n_no_reference": int((flagged["Alert"] == "No reference").sum())}
    if len(op):
        i, j = op["PredictedSales"].idxmax(), op["PredictedSales"].idxmin()
        out.update(avg_open=float(op["PredictedSales"].mean()),
                   peak_date=op.loc[i, "Date"], peak_value=float(op.loc[i, "PredictedSales"]),
                   trough_date=op.loc[j, "Date"], trough_value=float(op.loc[j, "PredictedSales"]))
    return out


def promo_split(flagged):
    """Descriptive split of predicted open-day sales by the Promo assumption (not a causal effect)."""
    op = flagged[flagged["Open"] == 1]
    a, b = op.loc[op["Promo"] == 1, "PredictedSales"], op.loc[op["Promo"] == 0, "PredictedSales"]
    if a.empty or b.empty:
        return None
    return {"promo_mean": float(a.mean()), "no_promo_mean": float(b.mean()),
            "n_promo": int(len(a)), "n_no_promo": int(len(b))}