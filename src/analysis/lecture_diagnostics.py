"""Diagnostic analysis of the Lecture building's zero-power behaviour (development window only).

Usage:  python -m src.analysis.lecture_diagnostics

Goal: describe the ~80 % zero-power readings without presuming whether they are real behaviour
or a metering artefact. Only data before the validation end date is used (test period untouched).
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config as C
from src.data import loaders as L
from src.visualization import style

B = "Lecture"


def runs(mask, ts):
    """Lengths (in minutes) of consecutive True runs; runs break across missing minutes."""
    m = np.asarray(mask, dtype=bool)
    contiguous = np.r_[False, np.diff(ts) == 60]
    # a new run starts when the value changes OR the minute sequence is broken
    change = np.r_[True, (m[1:] != m[:-1]) | ~contiguous[1:]]
    ids = np.cumsum(change)
    s = pd.Series(m).groupby(ids).agg(["first", "size"])
    return s.loc[s["first"], "size"].values, s.loc[~s["first"], "size"].values


def main():
    style.apply()
    split = json.load(open(C.MET_DIR / "split.json"))
    cut = pd.Timestamp(split["val_end"])
    df = L.load_energy_minutely(B, usecols=("timestamp", "power", "current", "voltage", "frequency", "power_factor"))
    t = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(seconds=C.IST_OFFSET_S)
    df = df[t < cut].dropna(subset=["power"]).copy()
    t = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(seconds=C.IST_OFFSET_S)
    df["zero"] = df["power"] == 0
    cal = L.load_calendar().set_index("date")
    d = t.dt.normalize()
    df["working_day"] = d.map(cal["working_day"]).values
    df["activity"] = d.map(cal["activity"]).values
    df["hour"], df["month"], df["dow"], df["year"] = t.dt.hour, t.dt.month, t.dt.dayofweek, t.dt.year

    out = {"window": f"{t.min()} to {t.max()} (development window, < {cut.date()})",
           "minutes_with_power": int(len(df)), "zero_minutes": int(df.zero.sum()),
           "zero_pct": round(100 * df.zero.mean(), 2)}
    # 1. by year, month, weekday, calendar flags
    tabs = {
        "by_year": df.groupby("year").zero.mean().mul(100).round(1),
        "by_month": df.groupby("month").zero.mean().mul(100).round(1),
        "by_dow": df.groupby("dow").zero.mean().mul(100).round(1),
        "by_working_day": df.groupby("working_day").zero.mean().mul(100).round(1),
        "by_activity": df.groupby("activity").zero.mean().mul(100).round(1),
    }
    for k, v in tabs.items():
        out[f"zero_pct_{k}"] = {str(a): float(b) for a, b in v.items()}
    # 2. run lengths
    ts = df["timestamp"].values
    zr, nr = runs(df.zero.values, ts)
    q = lambda a: dict(n=int(len(a)), median=float(np.median(a)), mean=round(float(a.mean()), 1),
                       p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)), max=int(a.max()))
    out["zero_run_minutes"], out["nonzero_run_minutes"] = q(zr), q(nr)
    out["zero_runs_longer_than_1_day"] = int((zr > 1440).sum())
    out["zero_minutes_in_runs_over_1_day_pct"] = round(100 * zr[zr > 1440].sum() / zr.sum(), 1)
    # 3. non-zero distribution
    nz = df.loc[~df.zero, "power"]
    out["nonzero_power_W"] = {k: round(float(v), 1) for k, v in nz.describe(percentiles=[.05, .25, .5, .75, .95]).items()}
    # 4. other channels during zero vs non-zero minutes
    chan = pd.DataFrame({
        "zero_minutes": [df.loc[df.zero, c].isna().mean() * 100 for c in ("current", "voltage", "frequency", "power_factor")],
        "nonzero_minutes": [df.loc[~df.zero, c].isna().mean() * 100 for c in ("current", "voltage", "frequency", "power_factor")],
    }, index=["current", "voltage", "frequency", "power_factor"]).round(1)
    out["pct_missing_by_channel"] = chan.to_dict()
    out["current_equals_0_in_zero_minutes_pct"] = round(float((df.loc[df.zero, "current"] == 0).sum() / df.loc[df.zero, "current"].notna().sum() * 100), 1)
    out["voltage_median_V_when_present_zero_vs_nonzero"] = [round(float(df.loc[df.zero, "voltage"].median()), 1), round(float(df.loc[~df.zero, "voltage"].median()), 1)]
    # 5. hourly view and occupancy while zero / mixed / nonzero
    base = pd.read_csv(C.PROC_DIR / "hourly_base.csv.gz", parse_dates=["timestamp"])
    h = base[(base.building == B) & (base.timestamp < cut) & (base.n_valid_min >= C.MIN_VALID_ENERGY_MIN)].copy()
    h["cls"] = np.where(h.n_zero_min == h.n_valid_min, "all-zero hour", np.where(h.n_zero_min == 0, "no-zero hour", "mixed hour"))
    out["hourly_class_pct"] = h.cls.value_counts(normalize=True).mul(100).round(1).to_dict()
    occ = h.dropna(subset=["occ_mean"]).groupby("cls").occ_mean.describe()[["count", "25%", "50%", "75%"]].round(1)
    out["occupancy_same_hour_by_class"] = occ.to_dict("index")
    out["hourly_target_zero_pct"] = round(100 * float((h.power_raw == 0).mean()), 2)
    h["hour"], h["month"] = h.timestamp.dt.hour, h.timestamp.dt.month
    pd.concat({k: v.rename("zero_pct") for k, v in tabs.items()}, names=["group", "level"]).to_csv(C.TAB_DIR / "lecture_zero_share_tables.csv")
    chan.to_csv(C.TAB_DIR / "lecture_channels_zero_vs_nonzero.csv")
    occ.to_csv(C.TAB_DIR / "lecture_occupancy_by_hour_class.csv")
    with open(C.MET_DIR / "lecture_diagnostics.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure (2 x 3)
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.4))
    hm = df.groupby(["month", "hour"]).zero.mean().unstack().mul(100)
    im = ax[0, 0].imshow(hm.values, aspect="auto", cmap="Greys", vmin=0, vmax=100, origin="lower")
    ax[0, 0].set_xticks(range(0, 24, 3)); ax[0, 0].set_yticks(range(12)); ax[0, 0].set_yticklabels(range(1, 13))
    ax[0, 0].set(title="Zero-power readings by month and hour", xlabel="hour of day (IST)", ylabel="month"); ax[0, 0].grid(False)
    fig.colorbar(im, ax=ax[0, 0], label="% of minutes at 0 W")
    bins = np.logspace(0, np.log10(max(zr.max(), nr.max()) + 1), 40)
    ax[0, 1].hist(zr, bins=bins, alpha=.7, color="#444444", label="zero runs")
    ax[0, 1].hist(nr, bins=bins, alpha=.6, color="#D55E00", label="non-zero runs")
    ax[0, 1].set(xscale="log", yscale="log", xlabel="run length (minutes)", ylabel="number of runs", title="Run lengths of zero / non-zero power")
    ax[0, 1].legend()
    ax[0, 2].hist(nz / 1000, bins=60, color="#0072B2")
    ax[0, 2].set(xlabel="power (kW), non-zero minutes only", ylabel="minutes", title="Distribution of non-zero power")
    wk = pd.DataFrame({"day of week": tabs["by_dow"].values}, index=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax[1, 0].bar(wk.index, wk["day of week"], color="#444444")
    ax[1, 0].set(ylabel="% of minutes at 0 W", title="Zero share by day of week", ylim=(0, 100))
    cats = ["working_day=1", "working_day=0", "activity H", "activity L"]
    vals = [tabs["by_working_day"].get(1), tabs["by_working_day"].get(0), tabs["by_activity"].get("H"), tabs["by_activity"].get("L")]
    ax[1, 1].bar(cats, vals, color=["#0072B2", "#0072B2", "#009E73", "#009E73"]); ax[1, 1].tick_params(axis="x", rotation=20)
    ax[1, 1].set(ylabel="% of minutes at 0 W", title="Zero share by academic calendar", ylim=(0, 100))
    order = ["no-zero hour", "mixed hour", "all-zero hour"]
    data = [h.loc[(h.cls == c) & h.occ_mean.notna(), "occ_mean"].values for c in order]
    ax[1, 2].boxplot(data, tick_labels=[f"{c}\n(n={len(d)})" for c, d in zip(order, data)], showfliers=False)
    ax[1, 2].set(ylabel="Lecture occupancy, same hour (count)", title="Occupancy vs zero-power class (exploratory)")
    fig.suptitle("Lecture building: zero-power diagnostics (development window)", y=1.0)
    fig.tight_layout()
    style.save(fig, "fig_lecture_zero_diagnostics.png")
    print(json.dumps(out, indent=1)[:4500])


if __name__ == "__main__":
    main()
