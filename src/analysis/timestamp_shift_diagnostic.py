"""Diagnostic: energy timestamps appear shifted by roughly 5 hours from July 2017 onward.

Usage:  python -m src.analysis.timestamp_shift_diagnostic     (after src.models.evaluate)

Evidence produced
  1. For each building and month, the circular shift (hours) that best aligns the monthly mean daily
     profile with a 2015-08..2016-06 reference profile (energy), and the same for occupancy.
  2. For six buildings, the shift of the energy timestamps (10-min steps) that maximises the Spearman
     correlation with occupancy before (2017-01-01..07-07) and after (2017-07-11..11-03) the break.
  3. Sensitivity of the FROZEN models' test metrics to the affected period (no refitting, no selection).
Nothing in the processed data or the models is modified by this script.
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config as C
from src.data import loaders as L
from src.models import common as M
from src.visualization import style

EPOCH = pd.Timestamp("1970-01-01")
BREAK_START, BREAK_END = pd.Timestamp("2017-07-08"), pd.Timestamp("2017-07-11")   # transition days excluded


def ep(ts):
    return int((ts - pd.Timedelta(seconds=C.IST_OFFSET_S) - EPOCH) // pd.Timedelta("1s"))


def monthly_profile_shift():
    base = pd.read_csv(C.PROC_DIR / "hourly_base.csv.gz", parse_dates=["timestamp"])
    base = base[base.power_w.notna()].copy()
    base["ym"], base["h"] = base.timestamp.dt.to_period("M"), base.timestamp.dt.hour
    out = {}
    for b, t in base.groupby("building"):
        ref = t[(t.timestamp >= "2015-08-01") & (t.timestamp < "2016-07-01")].groupby("h").power_w.mean().reindex(range(24))
        s = {}
        for ym, x in t.groupby("ym"):
            p = x.groupby("h").power_w.mean().reindex(range(24))
            if x.power_w.size < 240 or p.isna().any() or ref.isna().any() or p.std() == 0:
                continue
            s[str(ym)] = max(range(-12, 12), key=lambda k: np.corrcoef(np.roll(p.values, k), ref.values)[0, 1])
        out[b] = pd.Series(s)
    df = pd.DataFrame(out)
    df.to_csv(C.TAB_DIR / "timestamp_shift_monthly_profile_shift_hours.csv")
    return df


def occupancy_alignment():
    rows = []
    windows = {"pre (2017-01-01..07-07)": (ep(pd.Timestamp("2017-01-01")), ep(pd.Timestamp("2017-07-07"))),
               "post (2017-07-11..11-03)": (ep(BREAK_END), ep(pd.Timestamp("2017-11-03")))}
    for b in ["Academic", "Library", "Mess", "Lecture", "Boys_main", "Girls_main"]:
        e = L.load_energy_minutely(b).dropna(subset=["power"])
        e10 = e.groupby((e.timestamp // 600) * 600).power.mean()
        o = L.load_occupancy(b).set_index("timestamp").occupancy_count
        for name, (a, z) in windows.items():
            res = {}
            for s in range(-60, 61):
                ee = e10.copy(); ee.index = ee.index + s * 600
                j = pd.concat([ee.rename("p"), o.rename("o")], axis=1, join="inner")
                j = j[(j.index >= a) & (j.index <= z)]
                if len(j) >= 500:
                    res[s] = j.p.corr(j.o, method="spearman")
            s = max(res, key=res.get)
            rows.append(dict(building=b, period=name, best_energy_shift_minutes=s * 10, spearman_at_best=round(res[s], 3),
                             spearman_at_zero_shift=round(res[0], 3)))
    df = pd.DataFrame(rows)
    df.to_csv(C.TAB_DIR / "timestamp_shift_energy_vs_occupancy.csv", index=False)
    return df


def test_sensitivity():
    sigma = json.load(open(C.MET_DIR / "sigma_train_primary.json"))
    rows = []
    for v in ("P", "E_all", "baseline"):
        P = pd.read_csv(C.RESULTS / "predictions" / f"predictions_{v}.csv.gz", parse_dates=["timestamp"])
        P = P[P.split == "test"]
        if v == "E_all":
            P = P[P.common]                       # compare on the rows shared with the primary matrix
        periods = {"test before break (2017-01-01..07-07)": P.timestamp < BREAK_START,
                   "test after break (2017-07-11..11-04)": P.timestamp >= BREAK_END}
        for name, mask in periods.items():
            for (m, b), g in P[mask].groupby(["model", "building"]):
                rows.append(dict(variant=v, period=name, model=m, building=b, **M.metrics(g.y_true.values, g.y_pred.values, sigma[b])))
    df = pd.DataFrame(rows)
    df.to_csv(C.TAB_DIR / "test_sensitivity_to_timestamp_break_long.csv", index=False)
    macro = (df.groupby(["variant", "period", "model"]).agg(macro_R2=("R2", "mean"), macro_nMAE_std=("nMAE_std", "mean"),
                                                            macro_nRMSE_std=("nRMSE_std", "mean"), rows=("n", "sum")).reset_index())
    macro.to_csv(C.TAB_DIR / "test_sensitivity_to_timestamp_break_macro.csv", index=False)
    return df, macro


def figures():
    style.apply()
    base = pd.read_csv(C.PROC_DIR / "hourly_base.csv.gz", parse_dates=["timestamp"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    for a, b in zip(ax, ["Academic", "Library", "Mess"]):
        t = base[(base.building == b) & base.power_w.notna() & (base.timestamp.dt.dayofweek < 5)]
        for lab, lo, hi, c in (("Aug-Nov 2016", "2016-08-01", "2016-12-01", "#0072B2"), ("Jan-Jun 2017", "2017-01-01", "2017-07-01", "#009E73"),
                               ("Aug-Nov 2017", "2017-08-01", "2017-12-01", "#D55E00")):
            x = t[(t.timestamp >= lo) & (t.timestamp < hi)]
            a.plot(range(24), x.groupby(x.timestamp.dt.hour).power_w.mean() / 1000, color=c, label=lab)
        a.set(title=style.LABELS[b], xlabel="hour of day (as labelled, IST)", ylabel="mean weekday power (kW)")
        a.set_xticks(range(0, 24, 4))
    ax[0].legend()
    fig.suptitle("Mean weekday profile: the 2017-H2 profile is displaced by about 5 h", y=1.02)
    fig.tight_layout()
    style.save(fig, "fig19_timestamp_shift_profiles.png")


def main():
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    m = monthly_profile_shift()
    print(m.loc["2017-04":].T.to_string())
    print(occupancy_alignment().to_string(index=False))
    df, macro = test_sensitivity()
    print(macro.round(3).to_string(index=False))
    figures()


if __name__ == "__main__":
    main()
