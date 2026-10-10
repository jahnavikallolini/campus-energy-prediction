"""Exploratory data analysis on the DEVELOPMENT window (train + validation) only.

Usage:  python -m src.analysis.eda        (after build_features)

The test period is excluded from every statistic and figure here, except the coverage heat map
(coverage is a property of the raw data, not of model behaviour) which is labelled as such.
Occupancy of the *same* hour is shown for exploration only; it is not a model input.
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config as C
from src.features.build_features import PRIMARY_FEATURES, TARGET
from src.visualization import style

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load():
    split = json.load(open(C.MET_DIR / "split.json"))
    cut = pd.Timestamp(split["val_end"])
    base = pd.read_csv(C.PROC_DIR / "hourly_base.csv.gz", parse_dates=["timestamp"])
    feats = pd.read_csv(C.PROC_DIR / "hourly_features.csv.gz", parse_dates=["timestamp"])
    dev = feats[(feats.timestamp < cut) & feats[TARGET].notna()].copy()
    return split, cut, base, feats, dev


def lab(b):
    return style.LABELS[b]


def coverage_figure(base):
    b2 = base.copy()
    b2["ym"] = b2.timestamp.dt.to_period("M")
    m = b2.groupby(["building", "ym"]).power_w.apply(lambda s: 100 * s.notna().mean()).unstack(0)
    # months before a building's logging started are shown as blank
    first = base[base.n_valid_min > 0].groupby("building").timestamp.min()
    for b in m.columns:
        m.loc[m.index < first[b].to_period("M"), b] = np.nan
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(m.T.values, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_yticks(range(len(m.columns))); ax.set_yticklabels([lab(b) for b in m.columns])
    ticks = [i for i, p in enumerate(m.index) if p.month in (1, 7)]
    ax.set_xticks(ticks); ax.set_xticklabels([str(m.index[i]) for i in ticks], rotation=45, ha="right")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="% of hours with a valid reading")
    ax.set_title("Monthly coverage of valid hourly energy readings (full period, raw-data property)")
    style.save(fig, "fig01_coverage_by_month.png")
    m.round(1).to_csv(C.TAB_DIR / "coverage_by_month_pct.csv")


def summary_stats(dev):
    rows = []
    for b, t in dev.groupby("building", sort=False):
        p = t[TARGET]
        prof = t.groupby("hour")[TARGET].mean()
        wk = t[t.is_weekend == 0][TARGET].mean()
        we = t[t.is_weekend == 1][TARGET].mean()
        rows.append(dict(building=b, hours=len(p), mean_W=p.mean(), median_W=p.median(), std_W=p.std(),
                         cv=p.std() / p.mean(), min_W=p.min(), p05_W=p.quantile(.05), p95_W=p.quantile(.95),
                         max_W=p.max(), pct_hours_exactly_zero=100 * (p == 0).mean(),
                         peak_hour_of_mean_profile=int(prof.idxmax()), weekend_to_weekday_mean_ratio=we / wk))
    df = pd.DataFrame(rows).round(3)
    df.to_csv(C.TAB_DIR / "eda_summary_stats.csv", index=False)
    return df


def distribution_and_comparison(dev):
    fig, ax = plt.subplots(2, 4, figsize=(12, 5.4))
    for a, (b, t) in zip(ax.ravel(), dev.groupby("building", sort=False)):
        a.hist(t[TARGET] / 1000, bins=60, color=style.BUILDING_COLORS[b])
        a.set(title=lab(b), xlabel="hourly mean power (kW)", ylabel="hours")
    ax.ravel()[-1].axis("off")
    fig.suptitle("Distribution of hourly mean power by building (train + validation)", y=1.0)
    fig.tight_layout()
    style.save(fig, "fig02_power_distribution.png")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    order = list(C.BUILDINGS)
    data = [dev.loc[dev.building == b, TARGET] / 1000 for b in order]
    bp = ax.boxplot(data, orientation="horizontal", tick_labels=[lab(b) for b in order], showfliers=False,
                    patch_artist=True, medianprops=dict(color="black"))
    for patch, b in zip(bp["boxes"], order):
        patch.set_facecolor(style.BUILDING_COLORS[b]); patch.set_alpha(.7)
    ax.set(xlabel="hourly mean power (kW)", title="Building comparison: hourly power (whiskers 1.5 IQR, outliers hidden)")
    ax.invert_yaxis()
    style.save(fig, "fig06_building_comparison.png")


def hourly_profiles(dev):
    fig, ax = plt.subplots(2, 4, figsize=(12.5, 5.6), sharex=True)
    for a, (b, t) in zip(ax.ravel(), dev.groupby("building", sort=False)):
        for flag, ls, name in ((0, "-", "weekday"), (1, "--", "weekend")):
            g = t[t.is_weekend == flag].groupby("hour")[TARGET]
            a.plot(g.mean().index, g.mean() / 1000, ls, color=style.BUILDING_COLORS[b], label=name)
            a.fill_between(g.mean().index, g.quantile(.25) / 1000, g.quantile(.75) / 1000, alpha=.12,
                           color=style.BUILDING_COLORS[b])
        a.set(title=lab(b), xlabel="hour of day (IST)", ylabel="power (kW)")
        a.set_xticks(range(0, 24, 4))
    ax.ravel()[0].legend()
    ax.ravel()[-1].axis("off")
    fig.suptitle("Mean hourly profile, weekday vs weekend (band = interquartile range)", y=1.0)
    fig.tight_layout()
    style.save(fig, "fig03_hourly_profiles.png")


def weekly_monthly(dev):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    rows = []
    for b, t in dev.groupby("building", sort=False):
        mu = t[TARGET].mean()
        wk = t.groupby("dayofweek")[TARGET].mean() / mu
        mo = t.groupby("month")[TARGET].mean() / mu
        ax[0].plot(DOW, wk.values, marker="o", ms=3, color=style.BUILDING_COLORS[b], label=lab(b))
        ax[1].plot(mo.index, mo.values, marker="o", ms=3, color=style.BUILDING_COLORS[b])
        rows += [dict(building=b, kind="dayofweek", key=k, ratio=v) for k, v in wk.items()]
        rows += [dict(building=b, kind="month", key=k, ratio=v) for k, v in mo.items()]
    ax[0].set(title="Mean power by weekday (relative to building mean)", ylabel="ratio to building mean")
    ax[1].set(title="Mean power by month (relative to building mean)", xlabel="month", ylabel="ratio to building mean")
    ax[1].set_xticks(range(1, 13))
    ax[0].legend(ncol=2)
    fig.tight_layout()
    style.save(fig, "fig04_weekly_monthly_patterns.png")
    pd.DataFrame(rows).round(4).to_csv(C.TAB_DIR / "eda_weekly_monthly_ratios.csv", index=False)


def calendar_effect(dev):
    rows = []
    for b, t in dev.groupby("building", sort=False):
        mu = t[TARGET].mean()
        r = dict(building=b)
        for col, name in (("working_day", "working"), ("activity_high", "activity")):
            g = t.groupby(col)[TARGET].mean() / mu
            r[f"{name}_0_ratio"], r[f"{name}_1_ratio"] = g.get(0.0, np.nan), g.get(1.0, np.nan)
        rows.append(r)
    df = pd.DataFrame(rows).round(3)
    df.to_csv(C.TAB_DIR / "eda_calendar_effect.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    x = np.arange(len(df))
    ax[0].bar(x - .2, df.working_0_ratio, .4, label="non-working day", color="#999999")
    ax[0].bar(x + .2, df.working_1_ratio, .4, label="working day", color="#0072B2")
    ax[1].bar(x - .2, df.activity_0_ratio, .4, label="low activity (L)", color="#999999")
    ax[1].bar(x + .2, df.activity_1_ratio, .4, label="high activity (H)", color="#009E73")
    for a, t in zip(ax, ("Working vs non-working day", "Academic activity: low vs high")):
        a.set_xticks(x); a.set_xticklabels([lab(b) for b in df.building], rotation=30, ha="right")
        a.set(title=t); a.legend()
    ax[0].set_ylabel("mean power / building mean")
    fig.tight_layout()
    style.save(fig, "fig05_calendar_effect.png")
    return df


def occupancy_relationship(feats, dev):
    # (a) same-hour occupancy: exploratory only
    fig, ax = plt.subplots(2, 4, figsize=(12.5, 5.6))
    rows = []
    for a, (b, t) in zip(ax.ravel(), dev.groupby("building", sort=False)):
        d = t.dropna(subset=["occ_mean"])
        rho = d[TARGET].corr(d["occ_mean"], method="spearman")
        a.hexbin(d["occ_mean"], d[TARGET] / 1000, gridsize=35, cmap="Greys", mincnt=1, bins="log")
        a.set(title=f"{lab(b)}  (Spearman {rho:.2f})", xlabel="occupancy, same hour (count)", ylabel="power (kW)")
        rows.append(dict(building=b, n_hours=len(d), spearman_same_hour=rho,
                         pearson_same_hour=d[TARGET].corr(d["occ_mean"])))
    ax.ravel()[-1].axis("off")
    fig.suptitle("Hourly power vs same-hour occupancy (exploratory; not a model input)", y=1.0)
    fig.tight_layout()
    style.save(fig, "fig07_occupancy_vs_power.png")
    corr_same = pd.DataFrame(rows).round(3)
    # (b) lagged occupancy
    lags = [-3, -2, -1, 0, 1, 2, 3, 24, 48, 168]
    res = []
    cut = dev.timestamp.max()
    for b, t in feats[feats.timestamp <= cut].groupby("building", sort=False):
        t = t.sort_values("timestamp").reset_index(drop=True)
        for k in lags:
            o = t["occ_mean"].shift(k)   # k>0: occupancy k hours BEFORE the target hour; k<0: future (exploratory)
            d = pd.DataFrame({"p": t[TARGET], "o": o}).dropna()
            res.append(dict(building=b, lag_hours=k, spearman=d.p.corr(d.o, method="spearman"), n=len(d)))
    lagdf = pd.DataFrame(res).round(4)
    lagdf.to_csv(C.TAB_DIR / "eda_occupancy_lag_correlation.csv", index=False)
    corr_same.to_csv(C.TAB_DIR / "eda_occupancy_same_hour_corr.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for b in C.BUILDINGS:
        s = lagdf[lagdf.building == b].set_index("lag_hours").loc[[-3, -2, -1, 0, 1, 2, 3], "spearman"]
        ax.plot(s.index, s.values, marker="o", ms=3, color=style.BUILDING_COLORS[b], label=lab(b))
    ax.axvline(0, color="k", lw=.6)
    ax.set(xlabel="occupancy lag k (hours; k>0 = occupancy k h before the target hour, k<0 = after)",
           ylabel="Spearman correlation with power", title="Correlation of hourly power with shifted occupancy (exploratory)")
    ax.legend(ncol=2)
    style.save(fig, "fig08_lagged_occupancy_correlation.png")
    return corr_same, lagdf


def acf_figure(feats, dev):
    cut = dev.timestamp.max()
    rows = []
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for b, t in feats[feats.timestamp <= cut].groupby("building", sort=False):
        p = t.sort_values("timestamp")[TARGET].reset_index(drop=True)
        ac = [p.corr(p.shift(k)) for k in range(1, 169)]
        rows += [dict(building=b, lag=k, acf=a) for k, a in zip(range(1, 169), ac)]
        ax.plot(range(1, 169), ac, color=style.BUILDING_COLORS[b], lw=1, label=lab(b))
    for k in (24, 48, 72, 96, 120, 144, 168):
        ax.axvline(k, color="k", lw=.3, alpha=.4)
    ax.set(xlabel="lag (hours)", ylabel="autocorrelation (pairs where both hours valid)",
           title="Autocorrelation of hourly power (thin lines mark multiples of 24 h)")
    ax.legend(ncol=2, fontsize=7)
    style.save(fig, "fig09_autocorrelation.png")
    pd.DataFrame(rows).round(4).to_csv(C.TAB_DIR / "eda_autocorrelation.csv", index=False)


def feature_target_correlation(feats, cut):
    el = feats[(feats.timestamp < cut) & feats.eligible_primary]
    rows = {}
    for b, t in el.groupby("building", sort=False):
        rows[lab(b)] = {f: t[TARGET].corr(t[f], method="spearman") for f in PRIMARY_FEATURES}
    m = pd.DataFrame(rows)
    m.round(3).to_csv(C.TAB_DIR / "eda_feature_target_spearman.csv")
    fig, ax = plt.subplots(figsize=(8, 5.6))
    im = ax.imshow(m.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(m.columns, rotation=30, ha="right")
    ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(m.index); ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, label="Spearman correlation with hourly power")
    ax.set_title("Primary features vs target (train + validation, eligible rows)")
    style.save(fig, "fig10_feature_target_correlation.png")
    return m


def main():
    style.apply()
    split, cut, base, feats, dev = load()
    coverage_figure(base)
    stats = summary_stats(dev)
    distribution_and_comparison(dev)
    hourly_profiles(dev)
    weekly_monthly(dev)
    cal = calendar_effect(dev)
    corr_same, lagdf = occupancy_relationship(feats, dev)
    acf_figure(feats, dev)
    m = feature_target_correlation(feats, cut)
    pd.set_option("display.width", 220)
    print(stats.to_string()); print(cal.to_string()); print(corr_same.to_string())
    print(lagdf[lagdf.lag_hours.isin([-1, 0, 1, 2, 24])].pivot(index="building", columns="lag_hours", values="spearman").to_string())
    print(m.round(2).to_string())


if __name__ == "__main__":
    main()
