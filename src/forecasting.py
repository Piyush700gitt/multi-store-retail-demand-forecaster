"""Recursive multi-step sales forecasting (revenue). Shared by backtests and the Streamlit app.

history    : Store, Date, Sales, Open, is_trainable   (cleaned table)
store_meta : output of clean_store(), one row per Store
future_cal : Store, Date, Open, Promo, StateHoliday, SchoolHoliday
"""
import joblib
import numpy as np
import pandas as pd

from . import features as F

MAX_HORIZON = 30
MAX_GAP_DAYS = 14          # max days between last history date and forecast start
RECENT_WINDOW_DAYS = 56
MIN_RECENT_OBS = 14        # min open-day sales observations in that window
TAIL_DAYS = 100            # history actually needed (lags/rolling use <= 28 days)
CAL_COLS = ["Open", "Promo", "StateHoliday", "SchoolHoliday"]
BUNDLE_KEYS = ["model", "log_target", "store_cat", "features", "stores"]
OUT_COLS = ["Store", "Date", "Step", "DayOfWeek", "Open", "Promo", "StateHoliday",
            "SchoolHoliday", "PredictedSales", "ClippedToZero"]


class ForecastError(ValueError):
    """Invalid input or unusable history."""


def load_bundle(path):
    bundle = joblib.load(path)
    missing = [k for k in BUNDLE_KEYS if k not in bundle]
    if missing:
        raise ForecastError(f"Model bundle is missing keys: {missing}")
    return bundle


def latest_history_date(history):
    return history["Date"].max()


def default_future_calendar(history, stores, dates):
    """Fallback when the user supplies no calendar. Assumptions (document them!):
    Open = store's usual opening pattern for that weekday over the last 52 weeks;
    Promo = 0; no state or school holidays."""
    dates = pd.DatetimeIndex(dates)
    h = history[history["Store"].isin(stores)]
    recent = h[h["Date"] > h["Date"].max() - pd.Timedelta(days=364)]
    recent = recent.assign(dow=recent["Date"].dt.dayofweek)
    rate = recent.groupby(["Store", "dow"])["Open"].mean()
    glob = recent.groupby("dow")["Open"].mean()
    cal = pd.MultiIndex.from_product([stores, dates], names=["Store", "Date"]).to_frame(index=False)
    cal["dow"] = cal["Date"].dt.dayofweek
    rl = rate.rename("rate").reset_index()
    cal = cal.merge(rl, on=["Store", "dow"], how="left")
    cal["rate"] = cal["rate"].fillna(cal["dow"].map(glob)).fillna(1.0)
    cal["Open"] = (cal["rate"] >= 0.5).astype(int)
    cal["Promo"] = 0
    cal["StateHoliday"] = "0"
    cal["SchoolHoliday"] = 0
    return cal[["Store", "Date"] + CAL_COLS]


def _check_calendar(cal, stores, dates):
    need = ["Store", "Date"] + CAL_COLS
    absent = [c for c in need if c not in cal.columns]
    if absent:
        raise ForecastError(f"future_cal is missing columns: {absent}")
    cal = cal[need].copy()
    cal["Date"] = pd.to_datetime(cal["Date"])
    cal = cal[cal["Store"].isin(stores) & cal["Date"].isin(dates)]
    if cal.duplicated(["Store", "Date"]).any():
        raise ForecastError("future_cal has duplicate (Store, Date) rows.")
    if len(cal) != len(stores) * len(dates):
        raise ForecastError("future_cal must cover every store on every date from the day after "
                            "the last history date through the end of the horizon.")
    cal["StateHoliday"] = cal["StateHoliday"].astype(str)
    for c in ("Open", "Promo", "SchoolHoliday"):
        if not cal[c].isin([0, 1]).all():
            raise ForecastError(f"{c} must contain only 0/1.")
    if not cal["StateHoliday"].isin(F.STATE_HOLIDAY_CATS).all():
        raise ForecastError(f"StateHoliday must be one of {F.STATE_HOLIDAY_CATS}.")
    return cal


def forecast_stores(bundle, history, store_meta, stores, start_date, horizon=30, future_cal=None):
    """Recursive forecast for many stores at once (columns never mix -> no cross-store leakage).
    Returns (forecast_df, skipped) where skipped maps store -> reason."""
    start = pd.Timestamp(start_date).normalize()
    if not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= MAX_HORIZON:
        raise ForecastError(f"Horizon must be an integer between 1 and {MAX_HORIZON}.")
    stores = list(dict.fromkeys(int(s) for s in stores))
    if not stores:
        raise ForecastError("No stores requested.")
    known = set(store_meta["Store"]) & set(bundle["stores"])
    unknown = [s for s in stores if s not in known]
    if unknown:
        raise ForecastError(f"Unknown store ID(s): {unknown[:10]}")

    # INFORMATION BARRIER: nothing on or after the start date is ever read.
    past = history[history["Date"] < start]
    if past.empty:
        raise ForecastError("No history before the forecast start date.")
    last_hist = past["Date"].max()
    gap = (start - last_hist).days - 1
    if gap > MAX_GAP_DAYS:
        raise ForecastError(f"Start date is {gap} days past the latest history ({last_hist.date()}); "
                            f"the maximum supported gap is {MAX_GAP_DAYS} days.")
    end = start + pd.Timedelta(days=int(horizon) - 1)
    dates = pd.date_range(last_hist + pd.Timedelta(days=1), end)   # includes gap days, if any

    past = past[past["Store"].isin(stores)]
    recent = past[(past["Date"] > last_hist - pd.Timedelta(days=RECENT_WINDOW_DAYS))
                  & (past["is_trainable"] == 1)]
    n_recent = recent.groupby("Store").size().reindex(stores).fillna(0)
    skipped = {s: (f"Only {int(n)} open-day sales observations in the {RECENT_WINDOW_DAYS} days "
                   f"before the cutoff (need {MIN_RECENT_OBS}).")
               for s, n in n_recent.items() if n < MIN_RECENT_OBS}
    ok = [s for s in stores if s not in skipped]
    if not ok:
        return pd.DataFrame(columns=OUT_COLS), skipped

    cal = default_future_calendar(past, ok, dates) if future_cal is None \
        else _check_calendar(future_cal, ok, dates)
    cal = cal.merge(store_meta, on="Store", how="left", validate="many_to_one")
    cal = F.add_known_features(F.add_calendar_features(cal))

    tail_start = last_hist - pd.Timedelta(days=TAIL_DAYS)
    tail = past[past["Date"] > tail_start]
    wide = F.make_sales_wide(tail, date_index=pd.date_range(tail_start + pd.Timedelta(days=1), end))
    wide = wide.reindex(columns=ok)

    frames = []
    for t in dates:
        feats = F.compute_history_features(wide.loc[t - pd.Timedelta(days=60): t])
        hist_row = pd.DataFrame({n: f.loc[t] for n, f in feats.items()})
        X = cal[cal["Date"] == t].set_index("Store").join(hist_row).reset_index()
        Xm = X[bundle["features"]].copy()
        if bundle["store_cat"]:
            Xm["Store"] = pd.Categorical(Xm["Store"], categories=bundle["stores"])

        raw = bundle["model"].predict(Xm)
        if bundle["log_target"]:
            raw = np.expm1(raw)
        is_open = X["Open"].to_numpy() == 1
        clipped = raw < 0
        pred = np.where(is_open, np.clip(raw, 0, None), 0.0)       # closed day -> 0 revenue

        # feed the prediction back as "history" (NaN when closed, like in training)
        wide.loc[t, X["Store"].to_numpy()] = np.where(is_open & (pred > 0), pred, np.nan)

        if t >= start:
            frames.append(pd.DataFrame({
                "Store": X["Store"], "Date": t, "Step": (t - start).days + 1,
                "DayOfWeek": X["DayOfWeek"], "Open": X["Open"].astype(int),
                "Promo": X["Promo"].astype(int), "StateHoliday": X["StateHoliday"].astype(str),
                "SchoolHoliday": X["SchoolHoliday"].astype(int),
                "PredictedSales": pred, "ClippedToZero": clipped & is_open}))

    out = pd.concat(frames, ignore_index=True).sort_values(["Store", "Date"]).reset_index(drop=True)
    return out[OUT_COLS], skipped


def forecast_store(bundle, history, store_meta, store_id, start_date, horizon=30, future_cal=None):
    """Single-store convenience wrapper used by the app."""
    fc, skipped = forecast_stores(bundle, history, store_meta, [store_id], start_date, horizon, future_cal)
    if skipped:
        raise ForecastError(next(iter(skipped.values())))
    return fc
