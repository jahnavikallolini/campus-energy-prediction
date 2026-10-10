"""Stage-1 follow-up: is the post-break energy-vs-occupancy offset constant over time?

Usage: python -m src.analysis.timestamp_offset_by_month
For each building and month from 2017-01 to 2017-10, finds the shift of the energy timestamps (10-min steps,
-10 h .. +10 h) that maximises the Spearman correlation with occupancy. Estimation only: nothing is corrected.
"""
import numpy as np
import pandas as pd

from src import config as C
from src.data import loaders as L

EPOCH = pd.Timestamp("1970-01-01")


def ep(ts):
    return int((ts - pd.Timedelta(seconds=C.IST_OFFSET_S) - EPOCH) // pd.Timedelta("1s"))


def main():
    rows = []
    months = pd.period_range("2017-01", "2017-10", freq="M")
    for b in ["Academic", "Library", "Mess", "Boys_main", "Girls_main", "Facilities"]:
        e = L.load_energy_minutely(b).dropna(subset=["power"])
        e10 = e.groupby((e.timestamp // 600) * 600).power.mean()
        o = L.load_occupancy(b).set_index("timestamp").occupancy_count
        for m in months:
            a, z = ep(m.start_time), ep(m.end_time.floor("s"))
            res = {}
            for s in range(-60, 61):
                ee = e10.copy(); ee.index = ee.index + s * 600
                j = pd.concat([ee.rename("p"), o.rename("o")], axis=1, join="inner")
                j = j[(j.index >= a) & (j.index <= z)]
                if len(j) >= 300:
                    res[s] = j.p.corr(j.o, method="spearman")
            if not res:
                continue
            s = max(res, key=res.get)
            rows.append(dict(building=b, month=str(m), best_shift_min=s * 10, corr_best=round(res[s], 3),
                             corr_zero=round(res.get(0, np.nan), 3), corr_at_330=round(res.get(33, np.nan), 3)))
    df = pd.DataFrame(rows)
    df.to_csv(C.TAB_DIR / "timestamp_offset_by_month.csv", index=False)
    print(df.pivot(index="building", columns="month", values="best_shift_min").to_string())
    print(df.pivot(index="building", columns="month", values="corr_best").to_string())
    print(df.pivot(index="building", columns="month", values="corr_at_330").to_string())


if __name__ == "__main__":
    main()
