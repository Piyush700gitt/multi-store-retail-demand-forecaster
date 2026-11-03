"""Feature generation shared by training and the Streamlit app."""
import numpy as np
import pandas as pd

STATE_HOLIDAY_CATS = ["0", "a", "b", "c"]
STORE_TYPE_CATS = ["a", "b", "c", "d"]
ASSORTMENT_CATS = ["a", "b", "c"]

LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 28]

CALENDAR_FEATURES = ["Year", "Month", "Day", "DayOfWeek", "WeekOfYear", "DayOfYear",
                     "IsWeekend", "IsMonthStart", "IsMonthEnd"]
KNOWN_FEATURES = ["Store", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment",
                  "CompetitionDistance", "CompetitionDistance_missing", "CompetitionOpenMonths",
                  "Promo2", "Promo2Renewal"]
HISTORY_FEATURES = ["lag_1", "lag_7", "lag_14", "lag_28",
                    "roll_mean_7", "roll_mean_28", "roll_std_7", "roll_std_28",
                    "same_dow_mean_4"]
FEATURES = CALENDAR_FEATURES + KNOWN_FEATURES + HISTORY_FEATURES


# ---------- 1. Calendar features (known for any future date) ----------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = out["Date"]
    out["Year"] = d.dt.year
    out["Month"] = d.dt.month
    out["Day"] = d.dt.day
    out["DayOfWeek"] = d.dt.dayofweek + 1            # 1=Mon ... 7=Sun (Rossmann convention)
    out["WeekOfYear"] = d.dt.isocalendar().week.astype(int)
    out["DayOfYear"] = d.dt.dayofyear
    out["IsWeekend"] = (out["DayOfWeek"] >= 6).astype(int)
    out["IsMonthStart"] = d.dt.is_month_start.astype(int)
    out["IsMonthEnd"] = d.dt.is_month_end.astype(int)
    return out


# ---------- 2. Store / promo / holiday features (known in advance) ----------
def _to_cat(series, cats, name):
    s = series.astype(str)
    bad = set(s.unique()) - set(cats)
    if bad:
        raise ValueError(f"Unexpected {name} values: {sorted(bad)}")
    return pd.Categorical(s, categories=cats)


def _promo2_start(df):
    y, w = df["Promo2SinceYear"], df["Promo2SinceWeek"]
    ok = y.notna() & w.notna()
    start = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    txt = (y[ok].astype(int).astype(str) + "-" +
           w[ok].astype(int).astype(str).str.zfill(2) + "-1")
    start[ok] = pd.to_datetime(txt, format="%G-%V-%u", errors="coerce")
    return start


def add_known_features(df: pd.DataFrame) -> pd.DataFrame:
    """Requires: Date, Promo, StateHoliday, SchoolHoliday + cleaned store columns."""
    out = df.copy()
    out["StateHoliday"] = _to_cat(out["StateHoliday"], STATE_HOLIDAY_CATS, "StateHoliday")
    out["StoreType"] = _to_cat(out["StoreType"], STORE_TYPE_CATS, "StoreType")
    out["Assortment"] = _to_cat(out["Assortment"], ASSORTMENT_CATS, "Assortment")

    # Months since the nearest competitor opened (0 if not yet open; NaN if unknown)
    since = out["CompetitionOpenSince"]
    months = (out["Date"].dt.year - since.dt.year) * 12 + (out["Date"].dt.month - since.dt.month)
    out["CompetitionOpenMonths"] = months.clip(lower=0)

    # Promo2 renewal month: store is in Promo2, Promo2 has started, and this month
    # is one of the store's PromoInterval months (the months a new round begins).
    start = _promo2_start(out)
    mon = out["Date"].dt.strftime("%b")
    in_iv = np.fromiter((m in iv for m, iv in zip(mon, out["PromoInterval"].astype(str))),
                        dtype=bool, count=len(out))
    out["Promo2Renewal"] = ((out["Promo2"] == 1) & (out["Date"] >= start) & in_iv).astype(int)
    return out


# ---------- 3. History features (leak-safe, per store) ----------
def make_sales_wide(df: pd.DataFrame, date_index=None) -> pd.DataFrame:
    """Date x Store matrix of usable sales. Closed / open-zero / missing days are NaN.
    Pass date_index to extend the matrix into future dates (used by recursive forecasting)."""
    s = df["Sales"].where(df["is_trainable"] == 1)
    wide = (pd.DataFrame({"Date": df["Date"], "Store": df["Store"], "s": s})
            .pivot(index="Date", columns="Store", values="s"))
    if date_index is None:
        date_index = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    return wide.reindex(date_index).rename_axis("Date")


def compute_history_features(wide: pd.DataFrame) -> dict:
    """Each feature at date t uses only sales on dates < t. Columns (stores) never mix."""
    feats = {}
    for k in LAGS:
        feats[f"lag_{k}"] = wide.shift(k)
    shifted = wide.shift(1)
    for w in ROLL_WINDOWS:
        r = shifted.rolling(w, min_periods=max(2, w // 4))
        feats[f"roll_mean_{w}"] = r.mean()
        feats[f"roll_std_{w}"] = r.std()
    same_dow = [wide.shift(k) for k in (7, 14, 21, 28)]
    total = sum(x.fillna(0) for x in same_dow)
    count = sum(x.notna().astype(int) for x in same_dow)
    feats["same_dow_mean_4"] = total / count.where(count > 0)
    return feats


def build_feature_table(clean: pd.DataFrame) -> pd.DataFrame:
    """Training-time feature table: one row per (Store, Date) in `clean`."""
    df = add_known_features(add_calendar_features(clean))
    feats = compute_history_features(make_sales_wide(clean))
    hist = pd.concat([f.unstack().rename(n) for n, f in feats.items()], axis=1)  # index (Store, Date)
    hist = hist.reset_index()
    return df.merge(hist, on=["Store", "Date"], how="left", validate="one_to_one")
