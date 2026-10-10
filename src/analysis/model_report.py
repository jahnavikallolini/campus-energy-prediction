"""Figures and diagnostic tables for the regression experiments.

Usage:  python -m src.analysis.model_report        (after src.models.evaluate)
All numbers come from results/predictions/*.csv.gz and results/tables/model_metrics_long.csv.
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config as C
from src.models import common as M
from src.visualization import style

LAB = style.LABELS
MODEL_ORDER = [M.MODEL_NAMES[m] for m in M.MODELS]
MCOL = {"Linear Regression": "#0072B2", "KNN Regression": "#E69F00", "Decision Tree Regression": "#009E73",
        "Support Vector Regression": "#CC79A7"}


def load():
    long = pd.read_csv(C.TAB_DIR / "model_metrics_long.csv")
    best = json.load(open(C.MET_DIR / "best_model.json"))
    sigma = json.load(open(C.MET_DIR / "sigma_train_primary.json"))
    preds = {v: pd.read_csv(C.RESULTS / "predictions" / f"predictions_{v}.csv.gz", parse_dates=["timestamp"])
             for v in ("P", "E_matched", "E_all", "baseline")}
    feats = pd.read_csv(C.PROC_DIR / "hourly_features.csv.gz", parse_dates=["timestamp"],
                        usecols=["building", "timestamp", "hour", "dayofweek", "is_weekend", "activity_high", "working_day"])
    return long, best, sigma, preds, feats


def fig_val_test(long):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    d = long[(long.variant == "P") & (long.subset == "own")]
    bl = long[long.variant == "baseline"]
    x = np.arange(len(M.BUILDINGS))
    for a, split in zip(ax, ("val", "test")):
        for i, m in enumerate(MODEL_ORDER):
            v = d[(d.eval_set == split) & (d.model == m)].set_index("building").reindex(M.BUILDINGS).nRMSE_std
            a.bar(x + (i - 1.5) * 0.19, v.values, 0.19, color=MCOL[m], label=m)
        p = bl[(bl.eval_set == split) & bl.model.str.contains("persistence")].set_index("building").reindex(M.BUILDINGS).nRMSE_std
        a.plot(x, p.values, "k_", ms=14, mew=2, label="persistence baseline (t-1)")
        a.set_xticks(x); a.set_xticklabels([LAB[b] for b in M.BUILDINGS], rotation=25, ha="right")
        a.set_title(f"{'Validation' if split == 'val' else 'Test'}: standardised RMSE (RMSE / training std of target)")
    ax[0].set_ylabel("nRMSE_std (lower is better)")
    ax[0].legend(ncol=2, fontsize=7)
    fig.tight_layout()
    style.save(fig, "fig11_validation_vs_test_nrmse.png")


def representative_window(df_b, days=7):
    """First fully observed 7-day window in the test period after 3 April 2017, else first complete window."""
    s = df_b.set_index("timestamp").y_true
    idx = pd.date_range("2017-04-03", "2017-11-04", freq="D")
    for start in idx:
        w = s.loc[start: start + pd.Timedelta(days=days) - pd.Timedelta(hours=1)]
        if len(w) == 24 * days:
            return start
    return None


def fig_actual_pred(preds, best):
    P = preds["P"]
    pick = ["Academic", "Boys_main", "Library", "Lecture"]
    fig, ax = plt.subplots(len(pick), 1, figsize=(12, 9), sharex=False)
    rows = []
    for a, b in zip(ax, pick):
        g = P[(P.building == b) & (P.model == best["best_model"]) & (P.split == "test")]
        st = representative_window(g)
        if st is None:
            a.set_title(f"{LAB[b]}: no complete 7-day test window"); continue
        w = g[(g.timestamp >= st) & (g.timestamp < st + pd.Timedelta(days=7))]
        a.plot(w.timestamp, w.y_true / 1000, color="k", lw=1.4, label="actual")
        a.plot(w.timestamp, w.y_pred / 1000, color=MCOL[best["best_model"]], lw=1.2, label=f"{best['best_model']} (predicted)")
        a.set(ylabel="power (kW)", title=f"{LAB[b]}: test window starting {st.date()}")
        rows.append(dict(building=b, window_start=str(st.date())))
    ax[0].legend(ncol=2)
    fig.tight_layout()
    style.save(fig, "fig12_actual_vs_predicted.png")
    pd.DataFrame(rows).to_csv(C.TAB_DIR / "actual_vs_predicted_windows.csv", index=False)


def residual_tables(preds, best, sigma):
    P = preds["P"]
    g = P[P.model == best["best_model"]].copy()
    g["resid"] = g.y_true - g.y_pred
    rows = []
    for (b, s), x in g.groupby(["building", "split"]):
        r = x.resid
        rows.append(dict(building=b, split=s, n=len(r), bias_W=r.mean(), std_W=r.std(), skew=r.skew(),
                         q05_W=r.quantile(.05), q95_W=r.quantile(.95), std_over_sigma_train=r.std() / sigma[b],
                         pct_abs_resid_gt_2sigma_train=100 * (r.abs() > 2 * sigma[b]).mean()))
    t = pd.DataFrame(rows).round(3)
    t.to_csv(C.TAB_DIR / "residual_summary_best_model.csv", index=False)
    fig, ax = plt.subplots(2, 4, figsize=(13, 5.6))
    for a, b in zip(ax.ravel(), M.BUILDINGS):
        for s, c in (("val", "#0072B2"), ("test", "#D55E00")):
            r = g[(g.building == b) & (g.split == s)].resid / sigma[b]
            a.hist(r.clip(-4, 4), bins=60, range=(-4, 4), density=True, alpha=.55, color=c, label=s)
        a.axvline(0, color="k", lw=.6)
        a.set(title=LAB[b], xlabel="residual / training std (clipped at +-4)", ylabel="density")
    ax.ravel()[-1].axis("off"); ax[0, 0].legend()
    fig.suptitle(f"Residual distributions, {best['best_model']} (primary features)", y=1.0)
    fig.tight_layout()
    style.save(fig, "fig13_residual_distributions.png")
    return t


def error_breakdowns(preds, best, sigma, feats):
    g = preds["P"][preds["P"].model == best["best_model"]].merge(feats, on=["building", "timestamp"], how="left")
    g["abs_err_std"] = (g.y_true - g.y_pred).abs() / g.building.map(sigma)
    out = []
    for split in ("val", "test"):
        x = g[g.split == split]
        out.append(x.groupby(["building", "hour"]).abs_err_std.mean().rename("nMAE_std").reset_index().assign(split=split, group="hour", key=lambda d: d.hour)[["split", "group", "building", "key", "nMAE_std"]])
        for name, col, mp in (("day_type", "is_weekend", {0: "weekday", 1: "weekend"}), ("activity", "activity_high", {0.0: "L (low)", 1.0: "H (high)"}),
                              ("working_day", "working_day", {0.0: "non-working", 1.0: "working"})):
            out.append(x.groupby(["building", col]).abs_err_std.mean().rename("nMAE_std").reset_index().assign(split=split, group=name, key=lambda d: d[col].map(mp))[["split", "group", "building", "key", "nMAE_std"]])
    t = pd.concat(out, ignore_index=True)
    t.to_csv(C.TAB_DIR / "error_breakdown_best_model.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    for b in M.BUILDINGS:
        s = t[(t.split == "test") & (t.group == "hour") & (t.building == b)].sort_values("key")
        ax.plot(s.key, s.nMAE_std, color=style.BUILDING_COLORS[b], label=LAB[b])
    ax.set(xlabel="hour of day (IST)", ylabel="nMAE_std (MAE / training std)", title=f"Test error by hour of day, {best['best_model']}")
    ax.set_xticks(range(0, 24, 3)); ax.legend(ncol=2, fontsize=7)
    style.save(fig, "fig14_error_by_hour.png")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for a, grp, cats in zip(ax, ("day_type", "activity"), (["weekday", "weekend"], ["H (high)", "L (low)"])):
        x = np.arange(len(M.BUILDINGS))
        for i, c in enumerate(cats):
            v = t[(t.split == "test") & (t.group == grp) & (t.key == c)].set_index("building").reindex(M.BUILDINGS).nMAE_std
            a.bar(x + (i - .5) * .38, v.values, .38, label=c, color=["#0072B2", "#999999"][i])
        a.set_xticks(x); a.set_xticklabels([LAB[b] for b in M.BUILDINGS], rotation=30, ha="right"); a.legend()
        a.set_title({"day_type": "Test error: weekday vs weekend", "activity": "Test error: high vs low academic activity"}[grp])
    ax[0].set_ylabel("nMAE_std")
    fig.tight_layout()
    style.save(fig, "fig15_error_by_daytype_activity.png")
    return t


def primary_vs_energy_only(long, best):
    d = long[long.subset.isin(["own", "common"]) & long.variant.isin(["P", "E_matched", "E_all"])].copy()
    d["setup"] = np.select([d.variant == "P", d.variant == "E_matched", (d.variant == "E_all") & (d.subset == "common"),
                            (d.variant == "E_all") & (d.subset == "own")],
                           ["P: energy + lagged occupancy", "E_matched: energy-only, same train rows",
                            "E_all: energy-only, all train rows (scored on common rows)", "E_all: energy-only (scored on all its rows)"], "other")
    t = (d.groupby(["setup", "eval_set", "model"]).agg(macro_R2=("R2", "mean"), macro_nMAE_std=("nMAE_std", "mean"),
                                                         macro_nRMSE_std=("nRMSE_std", "mean"), rows=("n", "sum")).reset_index())
    t.to_csv(C.TAB_DIR / "primary_vs_energy_only_macro.csv", index=False)
    pb = d[d.model == best["best_model"]][["setup", "eval_set", "building", "MAE", "RMSE", "R2", "nRMSE_std", "n"]]
    pb.to_csv(C.TAB_DIR / "primary_vs_energy_only_best_model_by_building.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    setups = ["P: energy + lagged occupancy", "E_matched: energy-only, same train rows",
              "E_all: energy-only, all train rows (scored on common rows)"]
    cols = ["#0072B2", "#999999", "#D55E00"]
    x = np.arange(len(M.BUILDINGS))
    for a, split in zip(ax, ("val", "test")):
        for i, s in enumerate(setups):
            v = pb[(pb.setup == s) & (pb.eval_set == split)].set_index("building").reindex(M.BUILDINGS).nRMSE_std
            a.bar(x + (i - 1) * .27, v.values, .27, color=cols[i], label=s)
        a.set_xticks(x); a.set_xticklabels([LAB[b] for b in M.BUILDINGS], rotation=25, ha="right")
        a.set_title(f"{'Validation' if split == 'val' else 'Test'}, {best['best_model']}: nRMSE_std on common rows")
    ax[0].set_ylabel("nRMSE_std"); ax[0].legend(fontsize=7)
    fig.tight_layout()
    style.save(fig, "fig16_primary_vs_energy_only.png")
    return t, pb


def runtime_figure():
    t = pd.read_csv(C.TAB_DIR / "timings_and_clipping.csv")
    s = t[t.variant == "P"].groupby("model").agg(fit_seconds_total=("fit_seconds", "sum"), predict_seconds_total=("predict_seconds", "sum"),
                                                 clipped_val=("clipped_val", "sum"), clipped_test=("clipped_test", "sum")).reindex(MODEL_ORDER)
    s.round(2).to_csv(C.TAB_DIR / "runtime_summary_primary.csv")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar([m.replace(" Regression", "") for m in s.index], s.fit_seconds_total, color=[MCOL[m] for m in s.index])
    ax.set(yscale="log", ylabel="total training time, 7 buildings (s)", title="Training time (primary variant, 1 CPU core)")
    style.save(fig, "fig17_training_time.png")
    return s


def lecture_zero_analysis(preds, long):
    rows = []
    for v in ("P", "E_all"):
        for m, g in preds[v][(preds[v].building == "Lecture") & (preds[v].split == "test")].groupby("model"):
            z, nz = g[g.y_true == 0], g[g.y_true > 0]
            rows.append(dict(variant=v, model=m, split="test", n_zero_hours=len(z), pct_zero_hours=100 * len(z) / len(g),
                             MAE_zero_target_W=(z.y_pred - z.y_true).abs().mean(), MAE_nonzero_target_W=(nz.y_pred - nz.y_true).abs().mean(),
                             mean_pred_in_zero_hours_W=z.y_pred.mean(),
                             pct_zero_hours_pred_over_500W=100 * (z.y_pred > 500).mean(), pct_zero_hours_pred_exactly_0=100 * (z.y_pred == 0).mean(),
                             share_of_SSE_from_zero_hours=100 * ((z.y_pred - z.y_true) ** 2).sum() / ((g.y_pred - g.y_true) ** 2).sum()))
    t = pd.DataFrame(rows).round(2)
    t.to_csv(C.TAB_DIR / "lecture_zero_analysis_test.csv", index=False)
    return t


def fig_lecture(preds, best):
    g = preds["P"][(preds["P"].building == "Lecture") & (preds["P"].model == best["best_model"]) & (preds["P"].split == "test")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(g.y_true / 1000, g.y_pred / 1000, s=3, alpha=.3, color="#E69F00")
    mx = max(g.y_true.max(), g.y_pred.max()) / 1000
    ax[0].plot([0, mx], [0, mx], "k", lw=.6)
    ax[0].set(xlabel="actual (kW)", ylabel="predicted (kW)", title="Lecture, test: predicted vs actual")
    ax[1].boxplot([g[g.y_true == 0].y_pred / 1000, g[g.y_true > 0].y_pred / 1000], tick_labels=["actual = 0 W", "actual > 0 W"], showfliers=False)
    ax[1].set(ylabel="predicted (kW)", title="Predictions split by actual zero / non-zero")
    fig.tight_layout()
    style.save(fig, "fig18_lecture_predictions.png")


def main():
    style.apply()
    long, best, sigma, preds, feats = load()
    fig_val_test(long)
    fig_actual_pred(preds, best)
    res = residual_tables(preds, best, sigma)
    eb = error_breakdowns(preds, best, sigma, feats)
    pv, pb = primary_vs_energy_only(long, best)
    rt = runtime_figure()
    lz = lecture_zero_analysis(preds, long)
    fig_lecture(preds, best)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(res.to_string()); print(pv.round(3).to_string()); print(rt.to_string()); print(lz.to_string())


if __name__ == "__main__":
    main()
