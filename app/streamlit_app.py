"""Minimal Streamlit dashboard for the I-BLEND campus energy project.

Run from the project root:   streamlit run app/streamlit_app.py
Reads only saved outputs (results/anomalies, results/tables, data/processed) and the saved model files in models/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

MODELS = {"Support Vector Regression": "SVR (selected, primary)", "Linear Regression": "Linear Regression (sensitivity)"}
CODE = {"Support Vector Regression": "SVR", "Linear Regression": "LR"}
VARIANTS = {"P": "energy + lagged occupancy", "E_matched": "energy only (same rows)"}
BREAK = pd.Timestamp(C.PRIMARY_TEST_END_EXCL)
LABEL = {b: v["label"] for b, v in C.BUILDINGS.items()}

st.set_page_config(page_title="Campus energy: prediction and anomalies", layout="wide")


@st.cache_data
def load_flags():
    f = pd.read_csv(C.RESULTS / "anomalies" / "anomaly_flags.csv.gz", parse_dates=["timestamp"])
    return f


@st.cache_data
def load_tables():
    t = C.TAB_DIR
    return dict(period=pd.read_csv(t / "period_metrics_long.csv"), thr=pd.read_csv(t / "anomaly_thresholds.csv"),
                events=pd.read_csv(C.RESULTS / "anomalies" / "anomaly_events.csv", parse_dates=["start", "end"]))


def metrics(y, p):
    mse = mean_squared_error(y, p)
    return dict(MAE_W=mean_absolute_error(y, p), MSE_W2=mse, RMSE_W=float(np.sqrt(mse)), R2=r2_score(y, p) if len(y) > 1 else np.nan)


flags, tabs = load_flags(), load_tables()

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Selection")
b = st.sidebar.selectbox("Building (main feeder)", list(C.BUILDINGS), format_func=lambda x: LABEL[x])
model = st.sidebar.selectbox("Model", list(MODELS), format_func=lambda x: MODELS[x])
variant = st.sidebar.selectbox("Feature set", list(VARIANTS), format_func=lambda x: VARIANTS[x])
include_post = st.sidebar.checkbox("Include post-break rows (from 2017-07-08, timestamp issue)", value=False)
dmin, dmax = pd.Timestamp("2016-07-01"), pd.Timestamp("2017-11-04")
default = (pd.Timestamp(C.PRIMARY_TEST_START), pd.Timestamp("2017-07-07"))
rng = st.sidebar.date_input("Date range", value=(default[0].date(), default[1].date()), min_value=dmin.date(), max_value=dmax.date())
if not isinstance(rng, (list, tuple)) or len(rng) != 2:
    st.stop()
start, end = pd.Timestamp(rng[0]), pd.Timestamp(rng[1]) + pd.Timedelta(hours=23)
if not include_post:
    end = min(end, BREAK - pd.Timedelta(hours=1))

d = flags[(flags.building == b) & (flags.model == model) & (flags.variant == variant)]
d = d[(d.timestamp >= start) & (d.timestamp <= end)].sort_values("timestamp")

st.title("Campus energy consumption: prediction and potential anomalous periods")
st.caption("I-BLEND dataset, seven main feeders, hourly mean power in W. Periods: validation 2016-07-01 to 2016-12-31; "
           "primary test 2017-01-01 to 2017-07-07; post-break 2017-07-11 onward (sensitivity).")
st.warning("Energy timestamps appear displaced by about 5 h from 2017-07-08 onward (unresolved, not corrected). Results after that date are "
           "shown only as a sensitivity analysis. Flagged hours are statistical outliers relative to the model, not confirmed physical faults.")

if d.empty:
    st.info("No rows for this selection (the building may have no valid data in the chosen dates).")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Actual vs predicted", "Metrics", "Anomalies", "Export"])

with tab1:
    m = metrics(d.y_true, d.expected_W)
    c = st.columns(5)
    c[0].metric("Hours in selection", f"{len(d):,}")
    c[1].metric("MAE (W)", f"{m['MAE_W']:,.0f}")
    c[2].metric("RMSE (W)", f"{m['RMSE_W']:,.0f}")
    c[3].metric("R²", f"{m['R2']:.3f}")
    c[4].metric("Flagged hours", f"{int(d.flag.sum()):,} ({100 * d.flag.mean():.1f}%)")
    daily = d.set_index("timestamp").y_true.resample("D").mean() / 1000
    fig = go.Figure(go.Scatter(x=daily.index, y=daily.values, mode="lines", name="daily mean power"))
    fig.update_layout(height=320, yaxis_title="daily mean power (kW)", margin=dict(l=10, r=10, t=30, b=10), title=f"{LABEL[b]}: daily mean power")
    st.plotly_chart(fig, width="stretch")
    st.write(f"Model: **{MODELS[model]}**, features: **{VARIANTS[variant]}**. Expected consumption is the model prediction clipped at 0 W; "
             "metrics above are computed on the selected hours.")

with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.timestamp, y=d.y_true / 1000, mode="lines", name="actual", line=dict(color="black", width=1)))
    fig.add_trace(go.Scatter(x=d.timestamp, y=d.expected_W / 1000, mode="lines", name="expected (model)", line=dict(color="#CC79A7", width=1)))
    fl = d[d.flag]
    fig.add_trace(go.Scatter(x=fl.timestamp, y=fl.y_true / 1000, mode="markers", name="flagged hour",
                             marker=dict(size=5, color=np.where(fl.direction == "high", "#D55E00", "#0072B2"))))
    fig.update_layout(height=420, yaxis_title="power (kW)", margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Use the plot toolbar to zoom. Red markers: actual above expected; blue: below.")
    with st.expander("Recompute predictions with the saved model file"):
        if st.button("Run saved model on the selected hours"):
            import joblib
            from src.models import common as M
            feats = pd.read_csv(C.PROC_DIR / "hourly_features.csv.gz", parse_dates=["timestamp"])
            x = feats[(feats.building == b) & feats.timestamp.isin(d.timestamp)].sort_values("timestamp")
            pipe = joblib.load(ROOT / "models" / f"{variant}_{CODE[model]}_{b}.joblib")
            pred = np.clip(pipe.predict(x[M.VARIANTS[variant]["features"]]), 0, None)
            comp = pd.DataFrame({"timestamp": x.timestamp.values, "recomputed_W": pred}).merge(d[["timestamp", "expected_W"]], on="timestamp")
            st.write(f"Rows: {len(comp):,}; maximum absolute difference to the stored predictions: {np.abs(comp.recomputed_W - comp.expected_W).max():.3g} W (stored predictions are rounded to 6 significant digits)")

with tab3:
    st.subheader("Metrics for the current selection")
    sel = pd.DataFrame([metrics(d.y_true, d.expected_W)]).round(3)
    st.dataframe(sel, hide_index=True)
    st.subheader("Frozen-model metrics by evaluation period (per building)")
    p = tabs["period"]
    p = p[(p.variant == variant) & (p.building == b) & p.period.isin(["validation", "test_primary", "test_post_break", "test_full"])]
    p = p[["period", "model", "n", "MAE", "MSE", "RMSE", "R2", "nRMSE_std"]].round(3)
    st.dataframe(p.sort_values(["period", "model"]), hide_index=True)
    st.caption("nRMSE_std = RMSE divided by the building's training-target standard deviation. Models were fitted on 2014-02-17 to 2016-06-30 only.")
    st.subheader("Macro summary over buildings")
    mac = pd.read_csv(C.TAB_DIR / "period_metrics_macro.csv")
    mac = mac[(mac.variant == variant) & mac.period.isin(["validation", "test_primary", "test_post_break", "test_full"])]
    st.dataframe(mac[["period", "model", "rows", "macro_R2", "macro_nMAE_std", "macro_nRMSE_std"]].round(3).sort_values(["period", "model"]), hide_index=True)

with tab4:
    st.subheader("Flagged hours over time")
    wk = d.set_index("timestamp").flag.resample("W").sum()
    fig = go.Figure(go.Bar(x=wk.index, y=wk.values, marker_color="#D55E00"))
    fig.update_layout(height=280, yaxis_title="flagged hours per week", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    st.write(f"Rule: |modified z| > {C.ANOMALY_Z} on residual = actual - expected, with median/MAD taken from validation residuals "
             "per building, day-part and operating regime (see `src/anomaly/detect.py`).")
    ev = tabs["events"]
    ev = ev[(ev.building == b) & (ev.model == model) & (ev.variant == variant) & (ev.start >= start) & (ev.start <= end)]
    st.subheader(f"Events (consecutive flagged hours): {len(ev):,}")
    show = ev.sort_values("peak_abs_z", ascending=False)[["start", "end", "hours", "direction", "peak_abs_z", "excess_kWh"]].head(200)
    show = show.assign(peak_abs_z=show.peak_abs_z.round(2), excess_kWh=show.excess_kWh.round(2))
    st.dataframe(show, hide_index=True)
    with st.expander("Thresholds used (from validation residuals)"):
        th = tabs["thr"]
        st.dataframe(th[(th.building == b) & (th.model == model) & (th.variant == variant)][["daypart", "regime", "n_val", "median", "mad_used", "lower_resid_W", "upper_resid_W"]].round(1), hide_index=True)
    if b == "Lecture":
        st.info("Lecture is zero-inflated (about 80% of hours at 0 W); its residuals mix an off regime and an on regime, so its flags are less reliable than for other buildings.")

with tab5:
    out = d[["building", "timestamp", "period", "y_true", "expected_W", "resid", "z", "flag", "direction"]].rename(columns={"y_true": "actual_W", "resid": "residual_W", "z": "modified_z"})
    st.write(f"{len(out):,} rows for the current selection.")
    st.download_button("Download selection as CSV", out.to_csv(index=False).encode(), file_name=f"{b}_{CODE[model]}_{variant}_{start.date()}_{end.date()}.csv", mime="text/csv")
    ev_out = ev.copy()
    st.download_button("Download events as CSV", ev_out.to_csv(index=False).encode(), file_name=f"{b}_{CODE[model]}_{variant}_events.csv", mime="text/csv")
