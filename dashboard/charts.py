"""Plotly figure builders (presentation only)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ACTUAL, FORECAST, MUTED = "#4c78a8", "#e45756", "#9aa0a6"
ALERT_COLORS = {"High demand": "#2e7d32", "Low demand": "#c62828", "Normal": "#4c78a8",
                "No reference": "#90a4ae", "Closed": "#cfd8dc"}


def _base(fig, title, ytitle="Sales revenue", height=380):
    fig.update_layout(title=title, template="plotly_white", height=height,
                      margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2), yaxis_title=ytitle)
    return fig


def _vline(fig, x):
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper",
                  line=dict(color=MUTED, dash="dot", width=1.5))


def history_vs_forecast(hist_store, fc, start):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_store["Date"], y=hist_store["Sales"], mode="lines+markers",
                             name="Actual (open days)", line=dict(color=ACTUAL), marker=dict(size=4)))
    f = fc[fc["Open"] == 1]
    fig.add_trace(go.Scatter(x=f["Date"], y=f["PredictedSales"], mode="lines+markers",
                             name="Forecast (open days)", line=dict(color=FORECAST, dash="dash"),
                             marker=dict(size=5)))
    _vline(fig, pd.Timestamp(start).strftime("%Y-%m-%d"))
    return _base(fig, "Recent actual sales and forecast")


def daily_forecast_bars(flagged):
    fig = go.Figure()
    for label, color in ALERT_COLORS.items():
        d = flagged[flagged["Alert"] == label]
        if d.empty:
            continue
        fig.add_trace(go.Bar(x=d["Date"], y=d["PredictedSales"], name=label, marker_color=color,
                             hovertemplate="%{x|%a %d %b}<br>%{y:,.0f}<extra>" + label + "</extra>"))
    fig.update_layout(barmode="overlay")
    return _base(fig, "Predicted daily sales (colour = demand alert)")


def weekday_pattern(summary_row, fc):
    hist_vals = [summary_row.get(f"avg_dow_{i}", np.nan) for i in range(1, 8)]
    op = fc[fc["Open"] == 1]
    fc_vals = op.groupby("DayOfWeek")["PredictedSales"].mean().reindex(range(1, 8))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=DAY_LABELS, y=hist_vals, name="Historical average (open days)", marker_color=ACTUAL))
    fig.add_trace(go.Bar(x=DAY_LABELS, y=fc_vals.to_numpy(), name="Forecast average (open days)", marker_color=FORECAST))
    fig.update_layout(barmode="group")
    return _base(fig, "Weekly pattern: history vs forecast", height=340)


def store_position(summary, value):
    fig = go.Figure(go.Histogram(x=summary["avg_sales"], nbinsx=40, marker_color=ACTUAL))
    _vline(fig, float(value))
    fig.update_layout(showlegend=False)
    return _base(fig, "Average open-day sales across all stores (dotted line = selected store)",
                 ytitle="Number of stores", height=320)


def compare_weekday(summary, stores):
    s = summary.set_index("Store")
    fig = go.Figure()
    for sid in stores:
        y = [s.loc[sid].get(f"avg_dow_{i}", np.nan) for i in range(1, 8)]
        fig.add_trace(go.Scatter(x=DAY_LABELS, y=y, mode="lines+markers", name=f"Store {sid}"))
    return _base(fig, "Historical average sales by weekday")