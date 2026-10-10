"""Leakage-free feature engineering and chronological split for the hourly dataset.

Usage:  python -m src.features.build_features        (after src.data.preprocess)

Every feature for the hour that starts at time t uses only information available at t:
  * deterministic calendar terms of t (hour, weekday, month) and the published academic
    calendar for the date of t (working_day, activity: both fixed in advance by the institute);
  * target history: power at t-1, t-2, t-24 and rolling mean/std over t-24 ... t-1;
  * occupancy history: occupancy at t-1, t-2, t-24 (never occupancy of hour t itself).
Short gaps (<= MAX_FILL_HOURS) in the *inputs* are bridged by causal last-observation-carried-
forward; the target is never filled, and nothing is carried beyond a series' last observation.
"""
import json

import numpy as np
import pandas as pd

from src import config as C
from src.data import loaders as L

CALENDAR_FEATURES = ["hour", "dayofweek", "is_weekend", "month", "working_day", "activity_high"]
POWER_FEATURES = ["power_lag1", "power_lag2", "power_lag24", "power_roll_mean_24", "power_roll_std_24"]
OCC_FEATURES = ["occ_lag1", "occ_lag2", "occ_lag24"]
PRIMARY_FEATURES = CALENDAR_FEATURES + POWER_FEATURES + OCC_FEATURES
ENERGY_ONLY_FEATURES = CALENDAR_FEATURES + POWER_FEATURES
TARGET = "power_w"


def causal_fill(s: pd.Series, limit: int) -> pd.Series:
    """Carry the last observed value forward for at most `limit` consecutive missing hours,
    never beyond the last real observation of the series."""
    f = s.ffill(limit=limit)
    last = s.last_valid_index()
    if last is not None:
        f.loc[f.index > last] = np.nan
    return f


def building_features(t: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    t = t.sort_values("timestamp").reset_index(drop=True)
    G = C.MAX_FILL_HOURS
    p, o = t["power_w"], t["occ_mean"]
    pf, of = causal_fill(p, G), causal_fill(o, G)
    out = t[["building", "timestamp", TARGET, "n_valid_min", "n_zero_min", "occ_mean", "occ_n"]].copy()
    ts = t["timestamp"]
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["month"] = ts.dt.month
    dates = ts.dt.normalize()
    c = cal.set_index("date")
    out["working_day"] = dates.map(c["working_day"]).astype("float")
    out["activity_high"] = dates.map(c["activity"].eq("H").astype(float))
    for k in C.LAGS_POWER:
        out[f"power_lag{k}"] = pf.shift(k)
    shifted = pf.shift(1)
    out["power_roll_mean_24"] = shifted.rolling(C.ROLL_WINDOW, min_periods=C.ROLL_MIN_PERIODS).mean()
    out["power_roll_std_24"] = shifted.rolling(C.ROLL_WINDOW, min_periods=C.ROLL_MIN_PERIODS).std()
    for k in C.LAGS_OCC:
        out[f"occ_lag{k}"] = of.shift(k)
    # diagnostics: how many lag inputs came from carried-forward values
    filled = pd.DataFrame({f"p{k}": p.shift(k).isna() & pf.shift(k).notna() for k in C.LAGS_POWER})
    filled_o = pd.DataFrame({f"o{k}": o.shift(k).isna() & of.shift(k).notna() for k in C.LAGS_OCC})
    out["n_filled_inputs"] = filled.sum(axis=1) + filled_o.sum(axis=1)
    has_target = out[TARGET].notna()
    out["eligible_primary"] = has_target & out[PRIMARY_FEATURES].notna().all(axis=1)
    out["eligible_energy_only"] = has_target & out[ENERGY_ONLY_FEATURES].notna().all(axis=1)
    return out


def choose_split(feat: pd.DataFrame):
    """Pick global chronological split dates (month boundaries) by a fixed, documented rule.

    Among candidates whose pooled shares of eligible rows fall in the configured ranges, choose the
    one maximising the smallest per-building min(validation rows, test rows); ties go to the split
    closest to 70/15/15. Counts use rows eligible for the primary (occupancy-lagged) model.
    """
    el = feat[feat["eligible_primary"]].copy()
    el["ym"] = el["timestamp"].dt.to_period("M")
    months = pd.period_range(el["ym"].min(), el["ym"].max(), freq="M")
    cnt = el.pivot_table(index="ym", columns="building", values=TARGET, aggfunc="size", fill_value=0)
    cnt = cnt.reindex(months, fill_value=0)
    cum = cnt.cumsum()
    tot = cum.iloc[-1]
    pooled_total = float(tot.sum())
    cands = []
    for i in range(6, len(months) - 2):
        for j in range(i + 2, len(months)):
            tr = cum.iloc[i - 1]
            va = cum.iloc[j - 1] - tr
            te = tot - cum.iloc[j - 1]
            s_tr, s_va, s_te = tr.sum() / pooled_total, va.sum() / pooled_total, te.sum() / pooled_total
            if not (C.SPLIT_TRAIN_SHARE_RANGE[0] <= s_tr <= C.SPLIT_TRAIN_SHARE_RANGE[1]
                    and C.SPLIT_VAL_SHARE_RANGE[0] <= s_va <= C.SPLIT_VAL_SHARE_RANGE[1]
                    and C.SPLIT_TEST_SHARE_RANGE[0] <= s_te <= C.SPLIT_TEST_SHARE_RANGE[1]):
                continue
            score = float(np.minimum(va, te).min())
            dist = abs(s_tr - .70) + abs(s_va - .15) + abs(s_te - .15)
            cands.append((score, -dist, months[i], months[j], s_tr, s_va, s_te))
    if not cands:
        raise RuntimeError("no split satisfies the configured share ranges")
    cands.sort(reverse=True)
    best = cands[0]
    return dict(train_end=str((best[2]).to_timestamp().date()), val_end=str((best[3]).to_timestamp().date()),
                min_building_val_or_test_rows=int(best[0]),
                pooled_shares=dict(train=round(best[4], 3), val=round(best[5], 3), test=round(best[6], 3)),
                n_candidates=len(cands), top5=[(str(c[2]), str(c[3]), int(c[0])) for c in cands[:5]])


def assign_split(feat, split):
    te, ve = pd.Timestamp(split["train_end"]), pd.Timestamp(split["val_end"])
    s = np.where(feat["timestamp"] < te, "train", np.where(feat["timestamp"] < ve, "val", "test"))
    return pd.Series(s, index=feat.index)


def main():
    base = pd.read_csv(C.PROC_DIR / "hourly_base.csv.gz", parse_dates=["timestamp"])
    cal = L.load_calendar()
    feats = pd.concat([building_features(t, cal) for _, t in base.groupby("building", sort=False)],
                      ignore_index=True)
    assert feats["working_day"].notna().all(), "calendar does not cover every hour"
    split = choose_split(feats)
    feats["split"] = assign_split(feats, split)
    split.update(feature_set=PRIMARY_FEATURES, target=TARGET, target_unit=C.POWER_UNIT)
    with open(C.MET_DIR / "split.json", "w") as f:
        json.dump(split, f, indent=2)
    with open(C.MET_DIR / "feature_list.json", "w") as f:
        json.dump(dict(primary=PRIMARY_FEATURES, energy_only_ablation=ENERGY_ONLY_FEATURES, target=TARGET,
                       unit=C.POWER_UNIT, max_fill_hours=C.MAX_FILL_HOURS, roll_window=C.ROLL_WINDOW,
                       roll_min_periods=C.ROLL_MIN_PERIODS), f, indent=2)
    feats.to_csv(C.PROC_DIR / "hourly_features.csv.gz", index=False, float_format="%.6g")
    el = feats[feats["eligible_primary"]]
    tab = (el.pivot_table(index="building", columns="split", values=TARGET, aggfunc="size", fill_value=0)
           [["train", "val", "test"]])
    tab["total_primary"] = tab.sum(axis=1)
    print(json.dumps({k: v for k, v in split.items() if k not in ("feature_set",)}, indent=1))
    print(tab.to_string())
    print("rows (all hours, grid):", len(feats), "eligible primary:", int(feats.eligible_primary.sum()),
          "eligible energy-only:", int(feats.eligible_energy_only.sum()))


if __name__ == "__main__":
    main()
