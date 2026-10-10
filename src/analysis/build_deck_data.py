"""Collect the numbers and figure sizes used by presentation/build_deck.js from the saved result tables."""
import json

import pandas as pd
from PIL import Image

from src import config as C

SVR, LR, DT, KNN = "Support Vector Regression", "Linear Regression", "Decision Tree Regression", "KNN Regression"
PB, SN = "Baseline: persistence (t-1)", "Baseline: seasonal naive (t-24)"


def main():
    T = C.TAB_DIR
    pm = pd.read_csv(T / "period_metrics_macro.csv"); pl = pd.read_csv(T / "period_metrics_long.csv")
    cov = pd.read_csv(T / "coverage_summary.csv"); aso = pd.read_csv(T / "anomaly_summary_overall.csv")
    asb = pd.read_csv(T / "anomaly_summary_by_building_period.csv"); agr = pd.read_csv(T / "anomaly_agreement.csv")
    pk = json.load(open(C.MET_DIR / "pca_kmeans_summary.json")); lec = json.load(open(C.MET_DIR / "lecture_diagnostics.json"))
    run = pd.read_csv(T / "runtime_summary_primary.csv").set_index("model")
    M = lambda v, p, m, c="macro_nRMSE_std": float(pm[(pm.variant == v) & (pm.period == p) & (pm.model == m)][c].iloc[0])
    periods = ["validation", "test_primary", "test_post_break", "test_full"]
    d = dict(
        macro={p: {n: M("P" if "Baseline" not in m else "baseline", p, m) for n, m in (("SVR", SVR), ("Linear", LR), ("Decision Tree", DT), ("KNN", KNN), ("Persistence", PB))} for p in periods},
        macroR2={p: {n: M("P", p, m, "macro_R2") for n, m in (("SVR", SVR), ("Linear", LR), ("Decision Tree", DT), ("KNN", KNN))} for p in periods},
        fs={p: {n: {v: M(v, p, m) for v in ("P", "E_matched", "E_all")} for n, m in (("SVR", SVR), ("Linear", LR), ("Decision Tree", DT), ("KNN", KNN))} for p in ("validation", "test_primary")},
        perb={b: {p: {n: float(pl[(pl.variant == "P") & (pl.period == p) & (pl.model == m) & (pl.building == b)].R2.iloc[0]) for n, m in (("SVR", SVR), ("Linear", LR))} for p in ("validation", "test_primary")} for b in C.BUILDINGS},
        labels={b: v["label"] for b, v in C.BUILDINGS.items()},
        valid_hours={r.building: int(r.valid_hours) for r in cov.itertuples()}, primary_rows={r.building: int(r.primary_rows) for r in cov.itertuples()},
        anomaly={p: {"SVR_P": float(aso[(aso.model == SVR) & (aso.variant == "P") & (aso.period == p)].pct_flagged.iloc[0]),
                     "LR_P": float(aso[(aso.model == LR) & (aso.variant == "P") & (aso.period == p)].pct_flagged.iloc[0])} for p in ("validation", "test_primary", "test_post_break")},
        anomaly_test_count=int(aso[(aso.model == SVR) & (aso.variant == "P") & (aso.period == "test_primary")].flagged.iloc[0]),
        jaccard={c: float(agr[(agr.comparison == c) & (agr.period == "test_primary")].jaccard.iloc[0]) for c in agr.comparison.unique()},
        pca=pk["pca"], kmeans_all7=pk["kmeans_all7"], kmeans_wl=pk["kmeans_without_lecture"],
        lecture=dict(zero_pct=lec["zero_pct"], hourly_zero_pct=lec["hourly_target_zero_pct"]),
        rows=dict(train=90640, val=int(pm[(pm.variant == "P") & (pm.period == "validation")].rows.max()), test_primary=int(pm[(pm.variant == "P") & (pm.period == "test_primary")].rows.max()),
                  post=int(pm[(pm.variant == "P") & (pm.period == "test_post_break")].rows.max()), full=int(pm[(pm.variant == "P") & (pm.period == "test_full")].rows.max())),
        train_seconds=dict(SVR=float(run.loc[SVR, "fit_seconds_total"]), Linear=float(run.loc[LR, "fit_seconds_total"])),
    )
    sizes = {}
    for f in ["fig19_timestamp_shift_profiles.png", "fig21_anomaly_timeline.png", "fig22_anomaly_examples.png", "fig25_pca.png", "fig26_kmeans_without_lecture.png", "fig12_actual_vs_predicted.png", "fig20_anomaly_rates.png"]:
        w, h = Image.open(C.FIG_DIR / f).size
        sizes[f] = [w, h]
    d["fig_sizes"] = sizes
    json.dump(d, open(C.ROOT / "presentation" / "deck_data.json", "w"), indent=1)
    print("deck_data.json written")


if __name__ == "__main__":
    main()
