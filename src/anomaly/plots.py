"""Figures for the anomaly-detection stage.   Usage: python -m src.anomaly.plots  (after src.anomaly.detect)"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config as C
from src.visualization import style

SVR, LR = "Support Vector Regression", "Linear Regression"
BREAK = pd.Timestamp(C.PRIMARY_TEST_END_EXCL)


def load():
    f = pd.read_csv(C.RESULTS / "anomalies" / "anomaly_flags.csv.gz", parse_dates=["timestamp"])
    return f


def fig_rates(f):
    s = pd.read_csv(C.TAB_DIR / "anomaly_summary_by_building_period.csv")
    d = s[(s.model == SVR) & (s.variant == "P") & s.period.isin(["validation", "test_primary", "test_post_break"])]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.8))
    x = np.arange(len(C.BUILDINGS)); cols = {"validation": "#0072B2", "test_primary": "#009E73", "test_post_break": "#D55E00"}
    for i, p in enumerate(cols):
        v = d[d.period == p].set_index("building").reindex(list(C.BUILDINGS)).pct_flagged
        ax[0].bar(x + (i - 1) * .27, v.values, .27, color=cols[p], label=p.replace("_", " "))
    ax[0].set_xticks(x); ax[0].set_xticklabels([style.LABELS[b] for b in C.BUILDINGS], rotation=25, ha="right")
    ax[0].set(ylabel="% of hours flagged", title="SVR (primary features): flagged share by period"); ax[0].legend()
    o = pd.read_csv(C.TAB_DIR / "anomaly_summary_overall.csv")
    o = o[o.period.isin(["validation", "test_primary", "test_post_break"])]
    o["cfg"] = o.model.map({SVR: "SVR", LR: "LR"}) + " / " + o.variant.map({"P": "energy+occ", "E_matched": "energy-only"})
    for i, p in enumerate(cols):
        v = o[o.period == p].set_index("cfg").pct_flagged
        ax[1].bar(np.arange(len(v)) + (i - 1) * .27, v.values, .27, color=cols[p])
    ax[1].set_xticks(np.arange(len(v))); ax[1].set_xticklabels(v.index, rotation=20, ha="right")
    ax[1].set(ylabel="% of hours flagged (all buildings)", title="Flagged share by model and feature set")
    fig.tight_layout(); style.save(fig, "fig20_anomaly_rates.png")


def fig_timeline(f):
    d = f[(f.model == SVR) & (f.variant == "P")].copy()
    d["ym"] = d.timestamp.dt.to_period("M")
    m = d.groupby(["building", "ym"]).flag.mean().mul(100).unstack(0)[list(C.BUILDINGS)]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(m.T.values, aspect="auto", cmap="magma_r", vmin=0, vmax=30)
    ax.set_yticks(range(len(m.columns))); ax.set_yticklabels([style.LABELS[b] for b in m.columns])
    ax.set_xticks(range(len(m.index))); ax.set_xticklabels([str(p) for p in m.index], rotation=60, ha="right"); ax.grid(False)
    ib = list(m.index).index(pd.Period("2017-07", "M"))
    ax.axvline(ib - .5, color="cyan", lw=1.5)
    ax.axvline(list(m.index).index(pd.Period("2017-01", "M")) - .5, color="white", lw=1, ls="--")
    fig.colorbar(im, ax=ax, label="% of hours flagged")
    ax.set_title("Monthly share of flagged hours (SVR, energy + lagged occupancy). White dashed: test starts; cyan: timestamp break (~2017-07-08)", fontsize=9)
    style.save(fig, "fig21_anomaly_timeline.png")


def fig_windows(f):
    d = f[(f.model == SVR) & (f.variant == "P") & (f.period == "test_primary")]
    pick = ["Academic", "Library", "Boys_main", "Lecture"]
    fig, ax = plt.subplots(len(pick), 1, figsize=(12, 10))
    rows = []
    for a, b in zip(ax, pick):
        g = d[d.building == b].set_index("timestamp").sort_index()
        best, st = -1, None
        for s0 in pd.date_range(g.index.min().normalize(), g.index.max() - pd.Timedelta(days=7), freq="D"):
            w = g.loc[s0: s0 + pd.Timedelta(days=7) - pd.Timedelta(hours=1)]
            if len(w) == 168 and w.flag.sum() > best:
                best, st = int(w.flag.sum()), s0
        w = g.loc[st: st + pd.Timedelta(days=7) - pd.Timedelta(hours=1)]
        a.plot(w.index, w.y_true / 1000, "k", lw=1.2, label="actual")
        a.plot(w.index, w.expected_W / 1000, color="#CC79A7", lw=1.2, label="expected (SVR)")
        fl = w[w.flag]
        a.scatter(fl.index, fl.y_true / 1000, c=np.where(fl.direction == "high", "#D55E00", "#0072B2"), s=22, zorder=3)
        a.set(ylabel="power (kW)", title=f"{style.LABELS[b]}: {st.date()} to {(st + pd.Timedelta(days=6)).date()} ({best} flagged hours; red = high, blue = low)")
        rows.append(dict(building=b, window_start=str(st.date()), flagged_hours=best))
    ax[0].legend(ncol=2)
    fig.tight_layout(); style.save(fig, "fig22_anomaly_examples.png")
    pd.DataFrame(rows).to_csv(C.TAB_DIR / "anomaly_example_windows.csv", index=False)


def fig_resid(f):
    d = f[(f.model == SVR) & (f.variant == "P")]
    th = pd.read_csv(C.TAB_DIR / "anomaly_thresholds.csv")
    th = th[(th.model == SVR) & (th.variant == "P") & (th.daypart == "afternoon") & (th.regime == "normal")].set_index("building")
    fig, ax = plt.subplots(2, 4, figsize=(13, 5.6))
    for a, b in zip(ax.ravel(), C.BUILDINGS):
        g = d[d.building == b]
        sc = th.loc[b, "mad_used"] * 1.4826
        for per, c in (("validation", "#0072B2"), ("test_primary", "#009E73")):
            r = g[g.period == per].resid / 1000
            a.hist(r.clip(-8 * sc / 1000, 8 * sc / 1000), bins=60, range=(-8 * sc / 1000, 8 * sc / 1000), density=True, alpha=.55, color=c, label=per.replace("_", " "))
        for k in ("lower_resid_W", "upper_resid_W"):
            a.axvline(th.loc[b, k] / 1000, color="#D55E00", lw=1)
        a.set(title=style.LABELS[b], xlabel="residual = actual - expected (kW, clipped)", ylabel="density")
    ax.ravel()[-1].axis("off"); ax[0, 0].legend(fontsize=7)
    fig.suptitle("Residuals and thresholds (red lines: afternoon stratum, validation-derived)", y=1.0)
    fig.tight_layout(); style.save(fig, "fig23_residual_thresholds.png")


def fig_agreement():
    a = pd.read_csv(C.TAB_DIR / "anomaly_agreement.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    comps = list(a.comparison.unique()); pers = ["validation", "test_primary", "test_post_break"]
    for i, p in enumerate(pers):
        v = a[a.period == p].set_index("comparison").reindex(comps).jaccard
        ax.bar(np.arange(len(comps)) + (i - 1) * .27, v.values, .27, label=p.replace("_", " "))
    ax.set_xticks(range(len(comps))); ax.set_xticklabels(comps, rotation=15, ha="right", fontsize=8)
    ax.set(ylabel="Jaccard overlap of flagged hours", ylim=(0, 1), title="Agreement of anomaly flags on matched rows"); ax.legend()
    style.save(fig, "fig24_anomaly_agreement.png")


def main():
    style.apply(); f = load()
    fig_rates(f); fig_timeline(f); fig_windows(f); fig_resid(f); fig_agreement()
    print("anomaly figures written")


if __name__ == "__main__":
    main()
