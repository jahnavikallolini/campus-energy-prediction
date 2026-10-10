"""Validation-only selection of encoding and hyperparameters.

Usage:  python -m src.models.tune

For every feature-set variant (P = energy + lagged occupancy, E_all = energy-only on all eligible rows)
and every model, each candidate (encoding, hyperparameters) is fitted per building on TRAINING rows and
scored on VALIDATION rows. One configuration per (variant, model) is kept: the one with the lowest
macro-average (over buildings) validation standardised RMSE, i.e. RMSE divided by the building's
training-target standard deviation. The test period is never loaded here (load_variant(include_test=False)).
SVR candidates are fitted on a chronological stride subsample (<= 3,000 rows per building) for speed; the
selected SVR configuration is refitted on all training rows in evaluate.py.
"""
import json
import time

import numpy as np
import pandas as pd

from src import config as C
from src.models import common as M

SVR_TUNE_ROWS = 3000
TUNE_VARIANTS = ["P", "E_all"]


CKPT = C.RESULTS / "tuning_checkpoints"


def main():
    rows = []
    t0 = time.time()
    CKPT.mkdir(parents=True, exist_ok=True)
    for variant in TUNE_VARIANTS:
        data = M.load_variant(variant, include_test=False)          # train + val only
        F = M.VARIANTS[variant]["features"]
        for model in M.MODELS:
            ck = CKPT / f"{variant}_{model}.csv"
            if ck.exists():                                          # resume after an interrupted run
                rows += pd.read_csv(ck).to_dict("records")
                print(f"[{time.time() - t0:6.0f}s] loaded checkpoint {variant} / {model}", flush=True)
                continue
            part = []
            for enc, params in M.candidate_grid(model):
                for b in M.BUILDINGS:
                    tr = data["train"][data["train"].building == b]
                    va = data["val"][data["val"].building == b]
                    fit_rows = M.stride_subsample(tr, SVR_TUNE_ROWS) if model == "SVR" else tr
                    pipe = M.build_pipeline(model, params, enc, F)
                    pipe.fit(fit_rows[F], fit_rows[M.TARGET])
                    pred, _ = M.predict_clipped(pipe, va[F])
                    m = M.metrics(va[M.TARGET].values, pred, tr[M.TARGET].std())
                    part.append(dict(variant=variant, model=model, encoding=enc, params=json.dumps(params, default=str),
                                     building=b, **m))
            pd.DataFrame(part).to_csv(ck, index=False)
            rows += part
            print(f"[{time.time() - t0:6.0f}s] tuned {variant} / {model}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.TAB_DIR / "tuning_search_long.csv", index=False)
    agg = (df.groupby(["variant", "model", "encoding", "params"], sort=False)
             .agg(macro_val_nRMSE_std=("nRMSE_std", "mean"), macro_val_nMAE_std=("nMAE_std", "mean"),
                  macro_val_R2=("R2", "mean")).reset_index())
    agg.to_csv(C.TAB_DIR / "tuning_search_macro.csv", index=False)
    sel = {}
    for (variant, model), g in agg.groupby(["variant", "model"]):
        best = g.sort_values("macro_val_nRMSE_std").iloc[0]
        sel.setdefault(variant, {})[model] = dict(encoding=best.encoding, params=json.loads(best.params),
                                                  macro_val_nRMSE_std=float(best.macro_val_nRMSE_std),
                                                  macro_val_R2=float(best.macro_val_R2),
                                                  n_candidates=int(len(g)))
    sel["_meta"] = dict(criterion="lowest macro-average validation nRMSE_std (RMSE / training-target std)",
                        svr_tuning_subsample_rows=SVR_TUNE_ROWS, test_rows_used=False)
    with open(C.MET_DIR / "selected_config.json", "w") as f:
        json.dump(sel, f, indent=2)
    pd.set_option("display.width", 200)
    print(json.dumps({k: v for k, v in sel.items() if k != "_meta"}, indent=1))


if __name__ == "__main__":
    main()
