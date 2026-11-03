"""Streamlit dashboard. UI only: all forecasting logic lives in src/."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import charts                         # noqa: E402  (dashboard/charts.py)
from src import forecasting as FC     # noqa: E402
from src import insights as IN        # noqa: E402

st.set_page_config(page_title="Multi-Store Retail Sales Forecaster", page_icon="📈", layout="wide")

HOLIDAY_LABELS = {"0": "None", "a": "Public", "b": "Easter", "c": "Christmas"}
PROMO_CHOICES = {"No promotions": "none",
                 "Repeat this store's recent 14-day promo pattern": "repeat_14d"}


def fmt(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.0f}"


@st.cache_resource(show_spinner="Loading model and data...")
def load_artifacts():
    files = {"model": ROOT / "models" / "lgbm_final.joblib",
             "metrics": ROOT / "models" / "final_metrics.json",
             "card": ROOT / "models" / "model_card.json",
             "history": ROOT / "data" / "history.parquet",
             "meta": ROOT / "data" / "store_meta.parquet",
             "summary": ROOT / "data" / "store_summary.parquet"}
    missing = [str(p.relative_to(ROOT)) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing artifact file(s): " + ", ".join(missing))
    history = pd.read_parquet(files["history"])
    history["Store"] = history["Store"].astype("int64")
    history["StateHoliday"] = history["StateHoliday"].astype(str)
    return {"bundle": FC.load_bundle(files["model"]), "history": history,
            "meta": pd.read_parquet(files["meta"]), "summary": pd.read_parquet(files["summary"]),
            "metrics": json.loads(files["metrics"].read_text()),
            "card": json.loads(files["card"].read_text())}


try:
    art = load_artifacts()
except Exception as exc:  # show a readable message instead of a stack trace
    st.error(f"The app could not load its model files. {exc}")
    st.stop()

bundle, history, meta, summary = art["bundle"], art["history"], art["meta"], art["summary"]
metrics, card = art["metrics"], art["card"]
last_date = IN.last_history_date(history)
min_start, max_start = IN.allowed_start_range(history)
store_ids = sorted({int(s) for s in meta["Store"]} & {int(s) for s in bundle["stores"]})
meta_i = meta.set_index("Store")

st.title("Multi-Store Retail Sales Forecaster")
st.caption("Forecasts daily store **sales revenue** (not units) up to 30 days ahead.")

# ------------------------------- Sidebar -------------------------------
with st.sidebar:
    st.header("Forecast controls")
    store_id = st.selectbox("Store", store_ids,
                            format_func=lambda s: f"Store {s}  ·  type {meta_i.loc[s, 'StoreType']}")
    start_date = st.date_input("Forecast start date", value=min_start.date(),
                               min_value=min_start.date(), max_value=max_start.date(),
                               help="Forecasts start after the last date in the data. Later start dates "
                                    "need extra recursive steps, so errors can accumulate.")
    horizon = st.slider("Forecast horizon (days)", 1, FC.MAX_HORIZON, FC.MAX_HORIZON)
    promo_label = st.radio("Promotion assumption", list(PROMO_CHOICES),
                           help="Promotions must be known in advance. You can edit individual days in the Forecast tab.")
    with st.expander("Demand alert rules"):
        high = st.slider("High demand at or above (× usual weekday level)", 1.05, 2.00, IN.HIGH_DEFAULT, 0.05)
        low = st.slider("Low demand at or below (× usual weekday level)", 0.30, 0.95, IN.LOW_DEFAULT, 0.05)
    generate = st.button("Generate forecast", type="primary")
    st.caption(f"Latest data in the app: {last_date:%d %b %Y}")

start = pd.Timestamp(start_date)
promo_mode = PROMO_CHOICES[promo_label]

tab_over, tab_fc, tab_store, tab_perf, tab_ins = st.tabs(
    ["Overview", "Forecast", "Store comparison", "Model performance", "Insights & limits"])

# ------------------------------- Overview -------------------------------
with tab_over:
    st.subheader("What this app does")
    st.write("Estimates daily sales revenue for a chosen store for up to 30 days ahead and flags days that "
             "look unusually strong or weak compared with the store's usual weekday level. "
             "It forecasts **revenue**, not units, and does not compute inventory or reorder quantities.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "LightGBM")
    c2.metric("Stores supported", f"{len(store_ids):,}")
    c3.metric("Trained through", card["model_trained_through"])
    c4.metric("Input features", len(bundle["features"]))
    st.markdown(
        f"- **Method:** recursive forecasting. Each day's prediction feeds the next day's lag and rolling features.\n"
        f"- **Target:** {'log(1 + sales), converted back to sales' if bundle['log_target'] else 'sales'}; "
        f"{bundle['params'].get('n_estimators')} trees.\n"
        f"- **Data in the app:** {card['history_range'][0]} to {card['history_range'][1]} "
        f"(recent window; the model itself was trained on the full history).\n"
        f"- **How to use:** pick a store, start date and horizon in the sidebar, review the calendar in the "
        f"Forecast tab, then click **Generate forecast**.")

# ------------------------------- Forecast -------------------------------
with tab_fc:
    st.subheader("Calendar assumptions")
    st.caption("Open days, promotions and holidays must be known in advance. Defaults: usual weekday opening "
               "pattern, the promotion assumption from the sidebar, no state or school holidays. Edit any day.")
    try:
        cal_full = IN.build_calendar(history, store_id, start, horizon, promo_mode)
    except FC.ForecastError as exc:
        st.error(str(exc))
        cal_full = None

    future_cal = None
    if cal_full is not None:
        rows = cal_full[cal_full["Date"] >= start].reset_index(drop=True)
        view = pd.DataFrame({"Date": rows["Date"].dt.strftime("%a %d %b %Y"),
                             "Open": rows["Open"].astype(bool), "Promo": rows["Promo"].astype(bool),
                             "StateHoliday": rows["StateHoliday"].astype(str),
                             "SchoolHoliday": rows["SchoolHoliday"].astype(bool)})
        with st.expander("Review or edit the calendar for the forecast window"):
            edited = st.data_editor(
                view, key=f"cal|{store_id}|{start.date()}|{horizon}|{promo_mode}",
                hide_index=True, disabled=["Date"], num_rows="fixed",
                column_config={
                    "Open": st.column_config.CheckboxColumn("Open"),
                    "Promo": st.column_config.CheckboxColumn("Promo"),
                    "StateHoliday": st.column_config.SelectboxColumn(
                        "State holiday", options=list(HOLIDAY_LABELS), required=True,
                        help="0 = none, a = public holiday, b = Easter, c = Christmas"),
                    "SchoolHoliday": st.column_config.CheckboxColumn("School holiday")})

        def _flag(col):
            return edited[col].fillna(False).astype(bool).astype(int).to_numpy()

        edited_part = pd.DataFrame({"Store": int(store_id), "Date": rows["Date"].to_numpy(),
                                    "Open": _flag("Open"), "Promo": _flag("Promo"),
                                    "StateHoliday": edited["StateHoliday"].astype(str).to_numpy(),
                                    "SchoolHoliday": _flag("SchoolHoliday")})
        gap_part = cal_full.loc[cal_full["Date"] < start, ["Store", "Date", "Open", "Promo",
                                                           "StateHoliday", "SchoolHoliday"]]
        future_cal = pd.concat([gap_part, edited_part], ignore_index=True)

        if generate:
            with st.spinner("Generating forecast..."):
                try:
                    fc_new = FC.forecast_store(bundle, history, meta, store_id, start, horizon, future_cal)
                except FC.ForecastError as exc:
                    st.error(f"Could not generate a forecast: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected error while forecasting: {exc!r}")
                else:
                    st.session_state["result"] = {
                        "params": (store_id, start, horizon, promo_mode), "cal": future_cal,
                        "fc": fc_new, "ref": IN.reference_levels(history, store_id, start)}

    st.divider()
    res = st.session_state.get("result")
    if res is None:
        st.info("No forecast yet. Choose a store, start date and horizon in the sidebar, review the calendar "
                "above, then click **Generate forecast**.")
    else:
        r_store, r_start, r_h, _ = res["params"]
        if future_cal is None or res["params"] != (store_id, start, horizon, promo_mode) \
                or not res["cal"].equals(future_cal):
            st.warning(f"Inputs changed since this forecast was generated. Showing the previous result "
                       f"(Store {r_store}, from {r_start:%d %b %Y}). Click **Generate forecast** to update.")
        flagged = IN.flag_days(res["fc"], res["ref"], high, low)
        s = IN.summarize(flagged)
        recent = IN.recent_level(history, r_store, r_start)

        st.subheader(f"Store {r_store}: {r_h}-day forecast from {r_start:%d %b %Y}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total predicted sales", fmt(s["total"]))
        k2.metric("Average per open day", fmt(s["avg_open"]))
        k2.caption(f"{s['n_open']} open and {s['n_closed']} closed days")
        k3.metric("Highest day", fmt(s["peak_value"]))
        if s["peak_date"] is not None:
            k3.caption(f"{s['peak_date']:%a %d %b}")
        k4.metric("Lowest open day", fmt(s["trough_value"]))
        if s["trough_date"] is not None:
            k4.caption(f"{s['trough_date']:%a %d %b}")
        k5, k6, k7 = st.columns(3)
        k5.metric("High-demand days", s["n_high"])
        k6.metric("Low-demand days", s["n_low"])
        if pd.notna(recent) and pd.notna(s["avg_open"]) and recent > 0:
            k7.metric("Vs last 28 days (per open day)", f"{(s['avg_open'] / recent - 1) * 100:+.1f}%")
            k7.caption(f"Recent actual average: {fmt(recent)}")

        hist_store = history[(history["Store"] == r_store) & (history["is_trainable"] == 1)
                             & (history["Date"] > last_date - pd.Timedelta(days=90))].sort_values("Date")
        st.plotly_chart(charts.history_vs_forecast(hist_store, flagged, r_start), key="chart_hist")
        st.plotly_chart(charts.daily_forecast_bars(flagged), key="chart_bars")
        st.plotly_chart(charts.weekday_pattern(summary.set_index("Store").loc[r_store], flagged),
                        key="chart_weekday")

        st.markdown("#### Forecast table")
        ratio_txt = [f"{(r - 1) * 100:+.0f}%" if pd.notna(r) else "" for r in flagged["Ratio"]]
        table = pd.DataFrame({
            "Date": flagged["Date"].dt.date, "Weekday": flagged["Date"].dt.day_name(),
            "Store": flagged["Store"], "Predicted sales": flagged["PredictedSales"].round(0).astype(int),
            "Open": flagged["Open"].map({1: "Yes", 0: "No"}), "Promo": flagged["Promo"].map({1: "Yes", 0: "No"}),
            "State holiday": flagged["StateHoliday"].map(HOLIDAY_LABELS).fillna(flagged["StateHoliday"]),
            "School holiday": flagged["SchoolHoliday"].map({1: "Yes", 0: "No"}),
            "Vs usual weekday level": ratio_txt, "Alert": flagged["Alert"]})
        st.dataframe(table, hide_index=True)
        st.download_button("Download forecast as CSV", res["fc"].to_csv(index=False).encode("utf-8"),
                           file_name=f"forecast_store_{r_store}_{r_start:%Y%m%d}.csv", mime="text/csv")

# ------------------------------- Store comparison -------------------------------
with tab_store:
    st.subheader(f"Store {store_id} vs other stores")
    row = summary.set_index("Store").loc[store_id]
    rank_all = int((summary["avg_sales"] > row["avg_sales"]).sum()) + 1
    peers = summary[summary["StoreType"] == row["StoreType"]]
    rank_type = int((peers["avg_sales"] > row["avg_sales"]).sum()) + 1
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg sales per open day", fmt(row["avg_sales"]))
    m2.metric("Rank among all stores", f"{rank_all} of {len(summary)}")
    m3.metric(f"Rank in type {row['StoreType']}", f"{rank_type} of {len(peers)}")
    if pd.notna(row["avg_sales_promo"]) and pd.notna(row["avg_sales_no_promo"]) and row["avg_sales_no_promo"] > 0:
        m4.metric("Promo vs non-promo days", f"{(row['avg_sales_promo'] / row['avg_sales_no_promo'] - 1) * 100:+.0f}%")
        m4.caption("Historical, descriptive only")
    st.plotly_chart(charts.store_position(summary, row["avg_sales"]), key="chart_pos")
    extra = st.multiselect("Add stores to compare weekday patterns (up to 4)",
                           [s_ for s_ in store_ids if s_ != store_id], max_selections=4)
    st.plotly_chart(charts.compare_weekday(summary, [store_id] + extra), key="chart_cmp")
    st.markdown("#### Top 10 stores by average sales per open day")
    top = summary.sort_values("avg_sales", ascending=False).head(10)[["Store", "StoreType", "Assortment", "avg_sales"]]
    st.dataframe(top.rename(columns={"avg_sales": "Avg sales per open day"}).round({"Avg sales per open day": 0}),
                 hide_index=True)
    st.caption("Averages cover all open days in the training history.")

# ------------------------------- Model performance -------------------------------
with tab_perf:
    st.subheader("Backtest results (recursive 30-day forecasts, chronological holdout)")
    mo = metrics["overall"]
    m, b = mo["model"], mo["baseline_B"]
    p1, p2, p3 = st.columns(3)
    p1.metric("WAPE", f"{m['WAPE'] * 100:.2f}%", f"{(m['WAPE'] - b['WAPE']) * 100:+.2f} pp vs baseline",
              delta_color="inverse")
    p2.metric("MAE", f"{m['MAE']:,.0f}", f"{m['MAE'] - b['MAE']:+,.0f} vs baseline", delta_color="inverse")
    p3.metric("RMSE", f"{m['RMSE']:,.0f}", f"{m['RMSE'] - b['RMSE']:+,.0f} vs baseline", delta_color="inverse")
    st.caption("Baseline B = mean of the store's last 4 same-weekday sales before the cutoff. "
               "WAPE = total absolute error ÷ total actual sales. Scored on open days with sales > 0.")


    def metric_table(d, name):
        df = pd.DataFrame(d).T
        df.index.name = name
        return df.rename(columns={"n": "Store-days", "Model_WAPE_%": "Model WAPE %",
                                  "BaselineB_WAPE_%": "Baseline B WAPE %"})


    st.markdown("**Error by forecast horizon** (error can accumulate in recursive forecasts)")
    st.dataframe(metric_table(metrics["by_horizon"], "Days ahead"))
    t1, t2 = st.columns(2)
    t1.markdown("**By weekday** (1 = Monday)")
    t1.dataframe(metric_table(metrics["by_weekday"], "Weekday"))
    t2.markdown("**By promotion**")
    t2.dataframe(metric_table(metrics["by_promo"], "Promo"))
    st.markdown("**By store type**")
    st.dataframe(metric_table(metrics["by_store_type"], "Store type"))

    st.markdown("#### Methodology")
    meth = metrics["methodology"]
    st.markdown(
        f"- Evaluated model trained through **{meth['model_trained_through']}**; holdout "
        f"{meth['holdout_start']} to {meth['holdout_end']}.\n"
        f"- Forecast cutoffs: {', '.join(meth['forecast_cutoffs'])}; horizon {meth['horizon_days']} days.\n"
        f"- Backtests use the actual Open/Promo/holiday calendar; the app's defaults are less informed.\n"
        f"- {meth['scored_on']}.\n"
        f"- {card['evaluation_note']}")

# ------------------------------- Insights & limits -------------------------------
with tab_ins:
    st.subheader("Business insights")
    res = st.session_state.get("result")
    if res is None:
        st.info("Generate a forecast to see insights for the selected store.")
    else:
        r_store, r_start, r_h, _ = res["params"]
        flagged = IN.flag_days(res["fc"], res["ref"], high, low)
        s = IN.summarize(flagged)
        recent = IN.recent_level(history, r_store, r_start)
        pts = []
        if pd.notna(recent) and pd.notna(s["avg_open"]) and recent > 0:
            pts.append(f"Predicted average per open day is {fmt(s['avg_open'])}, "
                       f"{(s['avg_open'] / recent - 1) * 100:+.1f}% versus the store's last 28 days ({fmt(recent)}).")
        if s["peak_date"] is not None:
            pts.append(f"Highest predicted day: {s['peak_date']:%A %d %b} ({fmt(s['peak_value'])}); "
                       f"lowest open day: {s['trough_date']:%A %d %b} ({fmt(s['trough_value'])}).")
        ps = IN.promo_split(flagged)
        if ps:
            pts.append(f"Under your promo assumption, predicted open-day sales average {fmt(ps['promo_mean'])} on "
                       f"{ps['n_promo']} promo days vs {fmt(ps['no_promo_mean'])} on {ps['n_no_promo']} non-promo days. "
                       "This is the model's learned association, not a guaranteed uplift.")
        if s["n_closed"]:
            pts.append(f"{s['n_closed']} day(s) are assumed closed and counted as zero revenue.")
        pts.append(f"{s['n_high']} high-demand and {s['n_low']} low-demand day(s) under the current rules.")
        st.markdown("\n".join(f"- {p}" for p in pts))
        al = flagged[flagged["Alert"].isin(["High demand", "Low demand"])]
        if not al.empty:
            st.dataframe(pd.DataFrame({"Date": al["Date"].dt.date, "Weekday": al["Date"].dt.day_name(),
                                       "Predicted sales": al["PredictedSales"].round(0).astype(int),
                                       "Usual for weekday": al["Reference"].round(0).astype(int),
                                       "Alert": al["Alert"]}), hide_index=True)
    st.markdown("#### Alert rules")
    st.write(f"For each open day: ratio = predicted sales ÷ the store's median sales on that weekday over the last "
             f"{IN.REF_WINDOW_DAYS} days of data (at least {IN.MIN_REF_OBS} observations needed). "
             f"Ratio ≥ {high:.2f} is **High demand**; ratio ≤ {low:.2f} is **Low demand**. Flags are prompts for a "
             "manager's review, not stock or staffing instructions.")
    st.markdown("#### Limitations")
    st.markdown(
        "- Forecasts are **sales revenue**, not units. There is no stock, supplier or lead-time data, so nothing here is a reorder quantity.\n"
        "- Recursive forecasts feed on their own predictions, so errors can grow with the horizon (see Model performance).\n"
        "- Backtests cover one six-week summer window with overlapping forecasts; other seasons (such as December) are not evaluated.\n"
        "- Open days, promotions and holidays are assumptions you control. Defaults are simple, and wrong inputs give wrong forecasts.\n"
        "- Unplanned closures, local events, weather and competitor changes are not modelled.\n"
        "- Feature importance and promo comparisons describe associations in historical data, not causes.")