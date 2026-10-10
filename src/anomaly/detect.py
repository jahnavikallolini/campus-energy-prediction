"""Residual-based detection of potential anomalous consumption periods.

Usage:  python -m src.anomaly.detect        (after src.models.evaluate and src.analysis.test_periods)

Definition
  expected(t)  = clipped model prediction for the hour starting at t
  residual(t)  = actual(t) - expected(t)              [W]
  modified z   = 0.6745 * (residual - median_ref) / MAD_ref
  flag         = |modified z| > ANOMALY_Z (3.5, Iglewicz & Hoaglin 1993); sign gives high / low consumption
Reference statistics (median, MAD with a floor of 0.05 x the training-target std) come from VALIDATION residuals only (out-of-sample for the models, which are
fitted on training rows), computed separately for every model x feature set x building x day-part x regime stratum
(night 00-05, morning 06-11, afternoon 12-17, evening 18-23) and regime (expected power below 0.25 x training std = 'off-like', else normal) because residual spread depends on the time of day and on the operating regime.
Training residuals are not used: they are in-sample and understate the spread of future errors. The test
period never influences any threshold.
Models: SVR (selected, primary) and Linear Regression (sensitivity); feature sets: P (energy + lagged occupancy)
and E_matched (energy-only on the same rows), so the comparison is on matched observations.
Flagged hours are statistical outliers relative to the model, NOT confirmed physical faults.
"""
import json

import numpy as np
import pandas as pd

from src import config as C

MODELS = ["Support Vector Regression", "Linear Regression"]
VARIANTS = ["P", "E_matched"]
OUT_DIR = C.RESULTS / "anomalies"


def daypart(hour):
    out = pd.Series("night", index=hour.index)
    for name, (a, b) in C.DAYPARTS.items():
        out[(hour >= a) & (hour <= b)] = name
    return out


def period_of(ts, split):
    p = np.where(split == "val", "validation", "test_transition")
    p = np.where((split == "test") & (ts < pd.Timestamp(C.PRIMARY_TEST_END_EXCL)), "test_primary", p)
    p = np.where((split == "test") & (ts >= pd.Timestamp(C.POST_BREAK_START)), "test_post_break", p)
    return p


def reference_stats(df):
    """Median / MAD of validation residuals per (model, variant, building, daypart)."""
    v = df[df.split == "val"]
    g = v.groupby(["model", "variant", "building", "daypart", "regime"]).resid
    st = pd.DataFrame({"n_val": g.size(), "median": g.median(),
                       "mad": g.apply(lambda x: float(np.median(np.abs(x - np.median(x))))),
                       "std": g.std()}).reset_index()
    # Scale floor: in near-constant regimes (e.g. Lecture at 0 W) the stratum MAD collapses to a few watts and any
    # small deviation would score a huge z. The MAD is therefore not allowed to fall below a small fraction of the
    # building's training-target standard deviation (a deviation of a few percent of the building's normal
    # variability is not operationally meaningful). Only training data enter sigma_train.
    sigma = json.load(open(C.MET_DIR / "sigma_train_primary.json"))
    st["sigma_train"] = st["building"].map(sigma)
    st["mad_used"] = np.maximum(st["mad"], C.ANOMALY_SCALE_FLOOR_FRAC * st["sigma_train"])
    k = C.ANOMALY_Z / 0.6745
    st["lower_resid_W"] = st["median"] - k * st["mad_used"]
    st["upper_resid_W"] = st["median"] + k * st["mad_used"]
    return st


def events(flags):
    """Merge consecutive flagged hours (same sign) into events."""
    rows = []
    for (m, v, b), g in flags[flags.flag].groupby(["model", "variant", "building"]):
        g = g.sort_values("timestamp")
        new = (g.timestamp.diff() != pd.Timedelta(hours=1)) | (g.direction != g.direction.shift())
        for _, e in g.groupby(new.cumsum()):
            rows.append(dict(model=m, variant=v, building=b, start=e.timestamp.min(), end=e.timestamp.max(),
                             hours=len(e), direction=e.direction.iloc[0], peak_abs_z=e.z.abs().max(),
                             excess_kWh=e.resid.sum() / 1000.0, period=e.period.iloc[0]))
    return pd.DataFrame(rows)


def agreement(flags):
    rows = []
    def kappa(a, b):
        po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        return (po - pe) / (1 - pe) if pe < 1 else np.nan
    wide = flags.pivot_table(index=["building", "timestamp", "period"], columns=["model", "variant"], values="flag", aggfunc="first").reset_index()
    S, L_ = "Support Vector Regression", "Linear Regression"
    pairs = [("features: SVR P vs SVR E_matched", (S, "P"), (S, "E_matched")),
             ("features: LR P vs LR E_matched", (L_, "P"), (L_, "E_matched")),
             ("models: SVR P vs LR P", (S, "P"), (L_, "P"))]
    for period in ("validation", "test_primary", "test_post_break"):
        w = wide[wide.period == period]
        for name, a, b in pairs:
            x, y = w[a].astype(bool), w[b].astype(bool)
            both, either = (x & y).sum(), (x | y).sum()
            rows.append(dict(comparison=name, period=period, rows=len(w), flagged_a=int(x.sum()), flagged_b=int(y.sum()),
                             both=int(both), jaccard=both / either if either else np.nan, cohen_kappa=kappa(x, y)))
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for v in VARIANTS:
        P = pd.read_csv(C.RESULTS / "predictions" / f"predictions_{v}.csv.gz", parse_dates=["timestamp"])
        parts.append(P[P.model.isin(MODELS)])
    df = pd.concat(parts, ignore_index=True)
    df["resid"] = df.y_true - df.y_pred
    df["daypart"] = daypart(df.timestamp.dt.hour)
    sigma_all = json.load(open(C.MET_DIR / "sigma_train_primary.json"))
    # regime known at prediction time: expected power far below the building's usual variability (an "off-like" hour)
    df["regime"] = np.where(df.y_pred < C.ANOMALY_LOW_REGIME_FRAC * df.building.map(sigma_all), "low_expected", "normal")
    df["period"] = period_of(df.timestamp, df.split.values)
    ref = reference_stats(df)
    ref.to_csv(C.TAB_DIR / "anomaly_thresholds.csv", index=False)
    cols = ["median", "mad_used", "lower_resid_W", "upper_resid_W"]
    usable = ref[ref.n_val >= C.ANOMALY_MIN_STRATUM_ROWS]
    df = df.merge(usable[["model", "variant", "building", "daypart", "regime"] + cols], on=["model", "variant", "building", "daypart", "regime"], how="left")
    # strata with too few validation rows (e.g. a low-expected regime in a building that is never near zero) fall back to
    # the normal regime of the same day-part
    fb = ref[(ref.regime == "normal")][["model", "variant", "building", "daypart"] + cols].rename(columns={c: c + "_fb" for c in cols})
    df = df.merge(fb, on=["model", "variant", "building", "daypart"], how="left")
    for c in cols:
        df[c] = df[c].fillna(df[c + "_fb"])
    df["z"] = 0.6745 * (df.resid - df["median"]) / df["mad_used"]
    df["flag"] = df.z.abs() > C.ANOMALY_Z
    df["direction"] = np.where(~df.flag, "normal", np.where(df.z > 0, "high", "low"))
    df["expected_W"] = df.y_pred
    keep = ["model", "variant", "building", "timestamp", "split", "period", "daypart", "regime", "y_true", "expected_W", "resid", "z",
            "lower_resid_W", "upper_resid_W", "flag", "direction"]
    df[keep].to_csv(OUT_DIR / "anomaly_flags.csv.gz", index=False, float_format="%.5g")
    # summary by period (test_full = primary + transition + post-break)
    d2 = pd.concat([df, df[df.split == "test"].assign(period="test_full")], ignore_index=True)
    s = (d2.groupby(["model", "variant", "building", "period"]).agg(hours=("flag", "size"), flagged=("flag", "sum"),
                                                                   high=("direction", lambda x: (x == "high").sum()),
                                                                   low=("direction", lambda x: (x == "low").sum())).reset_index())
    s["pct_flagged"] = 100 * s.flagged / s.hours
    s.to_csv(C.TAB_DIR / "anomaly_summary_by_building_period.csv", index=False)
    tot = s.groupby(["model", "variant", "period"]).agg(hours=("hours", "sum"), flagged=("flagged", "sum"), high=("high", "sum"), low=("low", "sum")).reset_index()
    tot["pct_flagged"] = 100 * tot.flagged / tot.hours
    tot.to_csv(C.TAB_DIR / "anomaly_summary_overall.csv", index=False)
    ev = events(df)
    ev.to_csv(OUT_DIR / "anomaly_events.csv", index=False)
    sel = ev[(ev.model == MODELS[0]) & (ev.variant == "P") & (ev.period == "test_primary")].copy()
    sel["abs_excess_kWh"] = sel.excess_kWh.abs()
    # top 4 events per building by absolute energy deviation (peak z alone would favour Lecture's tiny residual scale)
    top = sel.sort_values("abs_excess_kWh", ascending=False).groupby("building").head(4).sort_values(["building", "abs_excess_kWh"], ascending=[True, False])
    top.to_csv(C.TAB_DIR / "top_anomaly_events_svr_primary_test.csv", index=False)
    ag = agreement(df)
    ag.to_csv(C.TAB_DIR / "anomaly_agreement.csv", index=False)
    meta = dict(criterion="|modified z| > %.1f" % C.ANOMALY_Z, reference="validation residuals only, per model x feature set x building x day-part",
                note="statistical outliers relative to the model, not confirmed physical faults")
    json.dump(meta, open(C.MET_DIR / "anomaly_method.json", "w"), indent=2)
    pd.set_option("display.width", 220)
    print(tot.round(2).to_string(index=False)); print(ag.round(3).to_string(index=False))
    print(s[(s.model == MODELS[0]) & (s.variant == "P")].pivot(index="building", columns="period", values="pct_flagged").round(1).to_string())
    print("events:", len(ev), "| top test_primary event:", top.iloc[0][["building", "start", "hours", "direction", "peak_abs_z"]].to_dict())


if __name__ == "__main__":
    main()
