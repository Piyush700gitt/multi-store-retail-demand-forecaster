"""Historical-cutoff evaluation of the recursive forecaster + seasonal-naive baseline."""
import numpy as np
import pandas as pd

from . import features as F
from . import forecasting as FC


def calendar_from_actuals(actual, stores, dates):
    """Known-in-advance calendar taken from the data (as Kaggle's test set provides it)."""
    dates = pd.DatetimeIndex(dates)
    cols = ["Open", "Promo", "StateHoliday", "SchoolHoliday"]
    sub = actual[actual["Date"].isin(dates) & actual["Store"].isin(stores)]
    idx = pd.MultiIndex.from_product([stores, dates], names=["Store", "Date"])
    cal = sub.drop_duplicates(["Store", "Date"]).set_index(["Store", "Date"])[cols].reindex(idx)
    cal["Open"] = cal["Open"].fillna(0).astype(int)          # missing rows -> treat as closed
    cal["Promo"] = cal["Promo"].fillna(0).astype(int)
    cal["SchoolHoliday"] = cal["SchoolHoliday"].fillna(0).astype(int)
    cal["StateHoliday"] = cal["StateHoliday"].fillna("0").astype(str)
    return cal.reset_index()


def seasonal_naive_forecast(history, start, dates, stores):
    """Baseline B: mean of the last 4 usable sales on the same weekday, using data before `start`."""
    start = pd.Timestamp(start)
    h = history[(history["Date"] < start) & history["Store"].isin(stores)]
    wide = F.make_sales_wide(h)
    dow_level = pd.DataFrame({d: wide[wide.index.dayofweek == d].tail(4).mean() for d in range(7)})
    fallback = wide.tail(56).mean()
    out = pd.MultiIndex.from_product([stores, pd.DatetimeIndex(dates)],
                                     names=["Store", "Date"]).to_frame(index=False)
    out["dow"] = out["Date"].dt.dayofweek
    lv = dow_level.stack().reset_index()
    lv.columns = ["Store", "dow", "Baseline"]
    out = out.merge(lv, on=["Store", "dow"], how="left")
    out["Baseline"] = out["Baseline"].fillna(out["Store"].map(fallback))
    return out[["Store", "Date", "Baseline"]]


def run_backtest(bundle, history, store_meta, first_start, offsets=(0, 4, 8, 12), horizon=30):
    """Forecast `horizon` days from several cutoffs; score against actual open-day sales."""
    stores_all = sorted(history["Store"].unique())
    frames, info = [], []
    for k in offsets:
        start = pd.Timestamp(first_start) + pd.Timedelta(days=int(k))
        dates = pd.date_range(start, periods=horizon)
        if dates[-1] > history["Date"].max():
            raise ValueError(f"Cutoff {start.date()} + {horizon} days exceeds available data.")
        cal = calendar_from_actuals(history, stores_all, dates)
        fc, skipped = FC.forecast_stores(bundle, history, store_meta, stores_all, start, horizon, cal)
        base = seasonal_naive_forecast(history, start, dates, sorted(fc["Store"].unique()))
        act = history[history["Date"].isin(dates) & (history["is_trainable"] == 1)][["Store", "Date", "Sales"]]
        m = (act.merge(fc[["Store", "Date", "Step", "DayOfWeek", "Promo", "PredictedSales"]],
                       on=["Store", "Date"], how="inner")
                .merge(base, on=["Store", "Date"], how="left"))
        m["Cutoff"] = start - pd.Timedelta(days=1)
        frames.append(m)
        info.append({"cutoff": (start - pd.Timedelta(days=1)).date(), "stores_forecast": fc["Store"].nunique(),
                     "stores_skipped": len(skipped), "rows_scored": len(m),
                     "clipped_to_zero": int(fc["ClippedToZero"].sum())})
    bt = pd.concat(frames, ignore_index=True).dropna(subset=["Baseline"])
    return bt, pd.DataFrame(info)
