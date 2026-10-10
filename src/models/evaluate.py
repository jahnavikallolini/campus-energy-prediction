"""Fit the frozen configurations, select the best model on VALIDATION, then score the test period.

Usage:  python -m src.models.evaluate        (after src.models.tune)

Phases
  A  fit every (variant, model, building) pipeline on TRAIN rows only; predict validation and test rows.
     No test metric is computed and no decision is taken in this phase.
  B  validation metrics -> best model chosen on validation (primary variant P) -> best_model.json written.
  C  test metrics are computed once for the frozen models and saved.
Baselines (persistence = power at t-1, seasonal naive = power at t-24) are reference rows, not candidates.
Predictions are clipped at 0 W; clipped counts are saved in timings_and_clipping.csv.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.models import common as M

EVAL_VARIANTS = ["P", "E_matched", "E_all"]
CKPT = C.RESULTS / "eval_checkpoints"
CFG_SOURCE = {"P": "P", "E_matched": "E_all", "E_all": "E_all"}   # energy-only variants share one tuned config


def macro(df):
    g = df.groupby(["variant", "eval_set", "subset", "model"], sort=False)
    out = g.agg(buildings=("building", "nunique"), macro_R2=("R2", "mean"), macro_nMAE_std=("nMAE_std", "mean"),
                macro_nRMSE_std=("nRMSE_std", "mean"), macro_RMSE_pct_of_mean=("RMSE_pct_of_mean", "mean")).reset_index()
    pooled = g.apply(lambda x: pd.Series(dict(pooled_MAE_W=np.average(x.MAE, weights=x.n),
                                              pooled_RMSE_W=np.sqrt(np.average(x.MSE, weights=x.n)))),
                     include_groups=False).reset_index()
    return out.merge(pooled, on=["variant", "eval_set", "subset", "model"])


def score(pred_df, sigma):
    rows = []
    for (variant, split, subset, model, b), g in pred_df.groupby(["variant", "split", "subset", "model", "building"], sort=False):
        rows.append(dict(variant=variant, eval_set=split, subset=subset, model=model, building=b,
                         **M.metrics(g.y_true.values, g.y_pred.values, sigma[b])))
    return pd.DataFrame(rows)


def main():
    cfg = json.load(open(C.MET_DIR / "selected_config.json"))
    CKPT.mkdir(parents=True, exist_ok=True)
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    preds, timing = [], []
    sigma = None
    # ---------------------------------------------------------------- phase A
    for variant in EVAL_VARIANTS:
        data = M.load_variant(variant, include_test=True)
        F = M.VARIANTS[variant]["features"]
        if variant == "P":
            sigma = {b: float(data["train"][data["train"].building == b][M.TARGET].std()) for b in M.BUILDINGS}
            base_rows = pd.concat([data["val"], data["test"]])
            for name, col in (("Baseline: persistence (t-1)", "power_lag1"), ("Baseline: seasonal naive (t-24)", "power_lag24")):
                preds.append(pd.DataFrame(dict(variant="baseline", model=name, building=base_rows.building, timestamp=base_rows.timestamp,
                                               split=np.where(base_rows.timestamp < M.split_dates()[1], "val", "test"),
                                               common=True, y_true=base_rows[M.TARGET], y_pred=base_rows[col])))
        for model in M.MODELS:
            c = cfg[CFG_SOURCE[variant]][model]
            for b in M.BUILDINGS:
                tr, va, te = (data[s][data[s].building == b] for s in ("train", "val", "test"))
                ck = CKPT / f"{variant}_{model}_{b}.joblib"
                if ck.exists():                                   # resume after an interrupted run
                    pv, nv, pt, nt, t_fit, t_pred = joblib.load(ck)
                else:
                    pipe = M.build_pipeline(model, c["params"], c["encoding"], F)
                    t_fit = M.fit_timed(pipe, tr[F], tr[M.TARGET])
                    t0 = time.perf_counter()
                    pv, nv = M.predict_clipped(pipe, va[F])
                    pt, nt = M.predict_clipped(pipe, te[F])
                    t_pred = time.perf_counter() - t0
                    joblib.dump(pipe, C.ROOT / "models" / f"{variant}_{model}_{b}.joblib", compress=3)
                    joblib.dump((pv, nv, pt, nt, t_fit, t_pred), ck)
                for split, d, p in (("val", va, pv), ("test", te, pt)):
                    preds.append(pd.DataFrame(dict(variant=variant, model=M.MODEL_NAMES[model], building=b, timestamp=d.timestamp.values,
                                                   split=split, common=d["common"].values, y_true=d[M.TARGET].values, y_pred=p)))
                timing.append(dict(variant=variant, model=M.MODEL_NAMES[model], building=b, train_rows=len(tr), fit_seconds=round(t_fit, 2),
                                   predict_seconds=round(t_pred, 3), clipped_val=nv, clipped_test=nt,
                                   encoding=c["encoding"], params=json.dumps(c["params"], default=str)))
                print(f"fitted {variant:9s} {model:3s} {b:10s} train_rows={len(tr):6d} fit={t_fit:6.1f}s", flush=True)
    P = pd.concat(preds, ignore_index=True)
    P["model"] = P["model"].replace({v: v for v in M.MODEL_NAMES.values()})
    pd.DataFrame(timing).to_csv(C.TAB_DIR / "timings_and_clipping.csv", index=False)
    for v, g in P.groupby("variant"):
        g.to_csv(C.RESULTS / "predictions" / f"predictions_{v}.csv.gz", index=False, float_format="%.6g")
    # expand to evaluation subsets: own rows, and common (rows also present in the primary matrix)
    own = P.assign(subset="own")
    com = P[P.common].assign(subset="common")
    allp = pd.concat([own, com[com.variant == "E_all"]], ignore_index=True)
    allp.loc[allp.variant.isin(["P", "E_matched", "baseline"]), "subset"] = "own"
    allp = allp.drop_duplicates(subset=["variant", "model", "building", "timestamp", "split", "subset"])
    # ---------------------------------------------------------------- phase B (validation only)
    val = score(allp[allp.split == "val"], sigma)
    val_p = val[(val.variant == "P") & (val.subset == "own")]
    summ = macro(val_p)
    best_row = summ.sort_values("macro_nRMSE_std").iloc[0]
    wins = val_p.loc[val_p.groupby("building").nRMSE_std.idxmin()].model.value_counts().to_dict()
    best = dict(best_model=best_row.model, criterion="lowest macro-average validation nRMSE_std on the primary (energy + lagged occupancy) variant",
                macro_val_nRMSE_std=float(best_row.macro_nRMSE_std), macro_val_R2=float(best_row.macro_R2),
                per_building_validation_wins=wins, decided_before_test_metrics=True,
                validation_ranking=summ.sort_values("macro_nRMSE_std")[["model", "macro_nRMSE_std", "macro_R2"]].to_dict("records"))
    with open(C.MET_DIR / "best_model.json", "w") as f:
        json.dump(best, f, indent=2)
    print("BEST (validation):", best["best_model"], flush=True)
    # ---------------------------------------------------------------- phase C (test, frozen models)
    test = score(allp[allp.split == "test"], sigma)
    long = pd.concat([val, test], ignore_index=True)
    long.to_csv(C.TAB_DIR / "model_metrics_long.csv", index=False)
    macro(long).to_csv(C.TAB_DIR / "model_metrics_macro.csv", index=False)
    json.dump(sigma, open(C.MET_DIR / "sigma_train_primary.json", "w"), indent=2)
    print(macro(long)[lambda d: d.variant.isin(["P", "baseline"]) & (d.subset == "own")].round(3).to_string())


if __name__ == "__main__":
    main()
