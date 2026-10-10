"""Metrics of the frozen models by evaluation period (validation, primary test, post-break, full test).

Usage:  python -m src.analysis.test_periods        (after src.models.evaluate)
No refitting: only saved predictions are re-aggregated. Periods are defined in src/config.py.
"""
import json

import pandas as pd

from src import config as C
from src.models import common as M

PERIODS = {
    "validation": lambda d: d.split == "val",
    "test_primary": lambda d: (d.split == "test") & (d.timestamp < pd.Timestamp(C.PRIMARY_TEST_END_EXCL)),
    "test_post_break": lambda d: (d.split == "test") & (d.timestamp >= pd.Timestamp(C.POST_BREAK_START)),
    "test_full": lambda d: d.split == "test",
}


def main():
    sigma = json.load(open(C.MET_DIR / "sigma_train_primary.json"))
    rows = []
    for v in ("P", "E_matched", "E_all", "baseline"):
        P = pd.read_csv(C.RESULTS / "predictions" / f"predictions_{v}.csv.gz", parse_dates=["timestamp"])
        if v == "E_all":
            P = P[P.common]                                  # matched rows only
        for name, f in PERIODS.items():
            for (m, b), g in P[f(P)].groupby(["model", "building"]):
                rows.append(dict(variant=v, period=name, model=m, building=b, **M.metrics(g.y_true.values, g.y_pred.values, sigma[b])))
    df = pd.DataFrame(rows)
    df.to_csv(C.TAB_DIR / "period_metrics_long.csv", index=False)
    mac = (df.groupby(["variant", "period", "model"]).agg(buildings=("building", "nunique"), rows=("n", "sum"), macro_R2=("R2", "mean"),
                                                          macro_nMAE_std=("nMAE_std", "mean"), macro_nRMSE_std=("nRMSE_std", "mean")).reset_index())
    mac.to_csv(C.TAB_DIR / "period_metrics_macro.csv", index=False)
    pd.set_option("display.width", 200)
    print(mac[mac.variant.isin(["P", "baseline"])].round(3).pivot_table(index="model", columns="period", values="macro_nRMSE_std").to_string())
    print(mac[mac.variant.isin(["P", "baseline"])].groupby("period").rows.max().to_dict())


if __name__ == "__main__":
    main()
