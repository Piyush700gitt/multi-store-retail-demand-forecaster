import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
from src import forecasting as FC, insights as IN

b = FC.load_bundle(ROOT / "models" / "lgbm_final.joblib")
h = pd.read_parquet(ROOT / "data" / "history.parquet")
h["Store"] = h["Store"].astype("int64"); h["StateHoliday"] = h["StateHoliday"].astype(str)
m = pd.read_parquet(ROOT / "data" / "store_meta.parquet")
lo, hi = IN.allowed_start_range(h)
store = int(sorted(set(h["Store"]) & set(m["Store"]))[0])

for mode in IN.PROMO_MODES:
    for start in (lo, hi):
        cal = IN.build_calendar(h, store, start, 30, mode)
        fc = FC.forecast_store(b, h, m, store, start, 30, cal)
        assert len(fc) == 30 and fc["Date"].min() == start and np.isfinite(fc["PredictedSales"]).all()
        fl = IN.flag_days(fc, IN.reference_levels(h, store, start))
        s = IN.summarize(fl)
        print(f"{mode:10s} start {start.date()}: total {s['total']:,.0f} | open {s['n_open']} | "
              f"high {s['n_high']} low {s['n_low']}")

for label, kw in [("start too late", dict(start_date=hi + pd.Timedelta(days=1), horizon=30)),
                  ("horizon 31", dict(start_date=lo, horizon=31)),
                  ("unknown store", dict(start_date=lo, horizon=30, store_id=99999))]:
    try:
        FC.forecast_store(b, h, m, kw.pop("store_id", store), **kw)
        raise AssertionError(f"{label} should have been rejected")
    except FC.ForecastError as e:
        print(f"rejected ({label}): {e}")
print("ALL CHECKS PASSED")