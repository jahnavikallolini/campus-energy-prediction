"""Hourly aggregation, coverage statistics and gap-handling experiments.

Usage:  python -m src.data.preprocess

Outputs
  data/processed/hourly_base.csv.gz     long table, one row per (building, hour) on the full hourly grid
  results/tables/threshold_sensitivity_energy.csv / _occupancy.csv
  results/tables/gap_runs_energy.csv, gap_run_distribution.csv
  results/tables/fill_experiment.csv, results/metrics/fill_experiment.json

Conventions
  * Hours are labelled by hour start in tz-naive IST wall-clock time.
  * Hourly energy = mean of valid one-minute `power` readings (no zero-filling) if at least
    MIN_VALID_ENERGY_MIN readings exist, otherwise missing.
  * Hourly occupancy = mean of the 10-minute counts present in the hour if at least
    MIN_VALID_OCC_OBS of 6 exist, otherwise missing. (The released 10-min counts are themselves
    maxima over 1-min windows, so this is a mean of maxima.)
  * Nothing is imputed here. Target values are never imputed anywhere in the project.
"""
import json

import numpy as np
import pandas as pd

from src import config as C
from src.data import loaders as L

GRID_START = pd.Timestamp("2013-08-10 00:00:00")
GRID_END = pd.Timestamp("2017-12-31 23:00:00")
HK0 = int((GRID_START - pd.Timestamp("1970-01-01")) // pd.Timedelta("1h"))
HK1 = int((GRID_END - pd.Timestamp("1970-01-01")) // pd.Timedelta("1h"))
GRID = pd.RangeIndex(HK0, HK1 + 1, name="hour_key")


def hourly_energy(building):
    df = L.load_energy_minutely(building).dropna(subset=["power"])
    hk = L.hour_key(df["timestamp"].values)
    g = df.assign(hk=hk, zero=(df["power"] == 0)).groupby("hk")
    out = pd.DataFrame({"power_raw": g["power"].mean(), "n_valid_min": g["power"].size(),
                        "n_zero_min": g["zero"].sum()})
    return out.reindex(GRID).fillna({"n_valid_min": 0, "n_zero_min": 0})


def hourly_occupancy(building):
    df = L.load_occupancy(building)
    hk = L.hour_key(df["timestamp"].values)
    g = df.assign(hk=hk).groupby("hk")["occupancy_count"]
    out = pd.DataFrame({"occ_raw": g.mean(), "occ_n": g.size()})
    return out.reindex(GRID).fillna({"occ_n": 0})


def build_base():
    parts = []
    for b in C.BUILDINGS:
        e, o = hourly_energy(b), hourly_occupancy(b)
        t = pd.concat([e, o], axis=1)
        t["power_w"] = t["power_raw"].where(t["n_valid_min"] >= C.MIN_VALID_ENERGY_MIN)
        t["occ_mean"] = t["occ_raw"].where(t["occ_n"] >= C.MIN_VALID_OCC_OBS)
        t.insert(0, "building", b)
        t.insert(1, "timestamp", L.hour_key_to_timestamp(t.index))
        parts.append(t.reset_index(drop=True))
        print(f"hourly built: {b}: valid energy hours {int(t.power_w.notna().sum())}, "
              f"valid occupancy hours {int(t.occ_mean.notna().sum())}")
    base = pd.concat(parts, ignore_index=True)
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    base.to_csv(C.PROC_DIR / "hourly_base.csv.gz", index=False, float_format="%.5g")
    return base


# ----------------------------------------------------------------- sensitivity
def span_bounds(t, col):
    idx = t.index[t[col] > 0]
    return idx.min(), idx.max()


def threshold_sensitivity(base):
    e_rows, o_rows = [], []
    for b, t in base.groupby("building", sort=False):
        t = t.reset_index(drop=True)
        s, e = span_bounds(t, "n_valid_min")
        span_h = e - s + 1
        for thr in C.ENERGY_THRESHOLD_GRID:
            n = int((t["n_valid_min"] >= thr).sum())
            e_rows.append(dict(building=b, min_valid_minutes=thr, valid_hours=n, span_hours=span_h,
                               pct_of_span=round(100 * n / span_h, 2)))
        so, eo = span_bounds(t, "occ_n")
        span_o = eo - so + 1
        for thr in C.OCC_THRESHOLD_GRID:
            n = int((t["occ_n"] >= thr).sum())
            joint = int(((t["occ_n"] >= thr) & (t["n_valid_min"] >= C.MIN_VALID_ENERGY_MIN)).sum())
            o_rows.append(dict(building=b, min_valid_obs=thr, valid_hours=n, span_hours=span_o,
                               pct_of_span=round(100 * n / span_o, 2),
                               hours_with_energy_and_occupancy=joint))
    ed, od = pd.DataFrame(e_rows), pd.DataFrame(o_rows)
    ed.to_csv(C.TAB_DIR / "threshold_sensitivity_energy.csv", index=False)
    od.to_csv(C.TAB_DIR / "threshold_sensitivity_occupancy.csv", index=False)
    return ed, od


# ------------------------------------------------------------------- gap runs
def invalid_runs(valid):
    """Return (start_index, length) of runs of False in a boolean array."""
    v = np.asarray(valid, dtype=bool)
    d = np.diff(np.r_[0, (~v).astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return starts, ends - starts


def gap_analysis(base):
    bins = [(1, 1), (2, 3), (4, 6), (7, 24), (25, 168), (169, 10**9)]
    labels = ["1 h", "2-3 h", "4-6 h", "7-24 h", "25 h-7 d", ">7 d"]
    dist, top = [], []
    for b, t in base.groupby("building", sort=False):
        t = t.reset_index(drop=True)
        s, e = span_bounds(t, "n_valid_min")
        valid = t["power_w"].notna().values[s:e + 1]
        st, ln = invalid_runs(valid)
        for (lo, hi), lab in zip(bins, labels):
            m = (ln >= lo) & (ln <= hi)
            dist.append(dict(building=b, gap_length=lab, n_gaps=int(m.sum()), hours_missing=int(ln[m].sum())))
        order = np.argsort(-ln)[:5]
        for i in order:
            top.append(dict(building=b, start=str(t["timestamp"].iloc[s + st[i]]), hours=int(ln[i]),
                            days=round(ln[i] / 24, 1)))
    dd, tt = pd.DataFrame(dist), pd.DataFrame(top)
    dd.to_csv(C.TAB_DIR / "gap_run_distribution.csv", index=False)
    tt.to_csv(C.TAB_DIR / "gap_runs_energy.csv", index=False)
    return dd, tt


# ----------------------------------------------------------- fill experiment
def fill_experiment(base, ks=(1, 2, 3, 6, 12, 24), n_samples=3000):
    """Mask artificial gaps in valid data and compare ways to estimate them.

    Methods: causal last-observation-carried-forward (LOCF, uses only the past),
    linear interpolation (uses the value *after* the gap, i.e. non-causal) and the
    seasonal-naive value from 24 h earlier. Only LOCF is leakage-free for building
    predictor inputs, so it is the method that is applied; linear is reported for reference.
    """
    rng = np.random.default_rng(C.SEED)
    rows = []
    for b, t in base.groupby("building", sort=False):
        p = t["power_w"].values.astype(float)
        v = ~np.isnan(p)
        cs = np.r_[0, np.cumsum(v)]
        n = len(p)
        mean_p = np.nanmean(p)
        for k in ks:
            i = np.arange(25, n - k - 1)
            ok = (cs[i + k] - cs[i] == k) & v[i - 1] & v[i + k] & (cs[i + k - 24] - cs[i - 24] == k)
            cand = i[ok]
            if len(cand) == 0:
                continue
            pick = rng.choice(cand, size=min(n_samples, len(cand)), replace=False)
            err = {"locf": [], "linear": [], "lag24": []}
            for s in pick:
                act = p[s:s + k]
                err["locf"].append(np.abs(act - p[s - 1]))
                lin = p[s - 1] + (p[s + k] - p[s - 1]) * (np.arange(1, k + 1) / (k + 1))
                err["linear"].append(np.abs(act - lin))
                err["lag24"].append(np.abs(act - p[s - 24:s + k - 24]))
            r = dict(building=b, gap_hours=k, n_samples=len(pick), mean_power_W=round(mean_p, 1))
            for m, a in err.items():
                r[f"mae_{m}_W"] = float(np.mean(np.concatenate(a)))
            r["locf_beats_lag24"] = r["mae_locf_W"] < r["mae_lag24_W"]
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(C.TAB_DIR / "fill_experiment.csv", index=False)
    ok = df.groupby("gap_hours")["locf_beats_lag24"].all()
    rec = max([k for k in ok.index if ok[k] and k <= 6], default=0)
    info = dict(rule="largest k <= 6 h for which causal LOCF has lower MAE than the 24 h-earlier value "
                     "in all seven buildings", all_buildings_ok_by_k=ok.to_dict(), recommended_max_fill_hours=int(rec),
                configured_MAX_FILL_HOURS=C.MAX_FILL_HOURS)
    with open(C.MET_DIR / "fill_experiment.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    return df, info


def main():
    for d in (C.TAB_DIR, C.MET_DIR, C.PROC_DIR):
        d.mkdir(parents=True, exist_ok=True)
    base = build_base()
    ed, od = threshold_sensitivity(base)
    print(ed.pivot(index="building", columns="min_valid_minutes", values="valid_hours").to_string())
    print(od.pivot(index="building", columns="min_valid_obs", values="valid_hours").to_string())
    dd, tt = gap_analysis(base)
    df, info = fill_experiment(base)
    print(json.dumps(info, indent=2, default=str))
    if info["recommended_max_fill_hours"] != C.MAX_FILL_HOURS:
        print("WARNING: configured MAX_FILL_HOURS differs from the experiment's recommendation")


if __name__ == "__main__":
    main()
