"""Reproducible dataset audit for the uploaded I-BLEND files.

Usage (from project root):  python -m src.data.audit
Writes tables to results/tables/audit_*.csv, a JSON summary to results/metrics/audit_summary.json
and a Markdown report to reports/dataset_audit.md. Raw data is only read, never modified.
"""
import json

import numpy as np
import pandas as pd

from src import config as C
from src.data import loaders as L

GRID_MINUTES = None  # filled from the status file (full 1-minute grid length)


def _ist(epoch):
    return pd.to_datetime(epoch, unit="s") + pd.Timedelta(seconds=C.IST_OFFSET_S)


def audit_energy_file(building, total_grid_minutes):
    df = pd.read_csv(L.energy_path(building), na_values=["NA"])
    ts = df["timestamp"].astype("int64")
    d = ts.diff().dropna()
    gaps = d[d > 60]
    p = df["power"]
    rec = dict(
        building=building, file=C.BUILDINGS[building]["energy"], rows=len(df), cols=df.shape[1],
        columns=",".join(df.columns),
        first_ts_ist=str(_ist(ts.iloc[0])), last_ts_ist=str(_ist(ts.iloc[-1])),
        monotonic=bool(ts.is_monotonic_increasing), dup_timestamps=int(ts.duplicated().sum()),
        dup_rows=int(df.duplicated().sum()), modal_step_s=int(d.mode().iloc[0]),
        steps_below_60s=int((d < 60).sum()), n_gaps_over_60s=int(len(gaps)),
        absent_minutes_within_span=int((gaps / 60 - 1).sum()),
        longest_gap_days=round(float(d.max()) / 86400, 2),
        power_min=p.min(), power_median=p.median(), power_mean=p.mean(), power_max=p.max(),
        power_zero_rows=int((p == 0).sum()), power_negative_rows=int((p < 0).sum()),
        share_of_full_grid_pct=round(100 * p.notna().sum() / total_grid_minutes, 2),
    )
    for c in df.columns:
        rec[f"na_{c}"] = int(df[c].isna().sum())
    first_cur = ts[df["current"].notna()]
    rec["current_first_valid_ist"] = str(_ist(first_cur.iloc[0])) if len(first_cur) else None
    return rec, df[["timestamp", "power"]].set_index(ts.values)["power"]


def audit_occupancy_file(building):
    df = L.load_occupancy(building)
    ts, o = df["timestamp"], df["occupancy_count"]
    d = ts.diff().dropna()
    return dict(
        building=building, occ_code=C.BUILDINGS[building]["occ"], rows=len(df),
        first_ts_ist=str(_ist(ts.iloc[0])), last_ts_ist=str(_ist(ts.iloc[-1])),
        monotonic=bool(ts.is_monotonic_increasing), dup_timestamps=int(ts.duplicated().sum()),
        dup_rows=int(df.duplicated().sum()), modal_step_s=int(d.mode().iloc[0]),
        off_grid_600s=int((ts % 600 != 0).sum()), n_gaps_over_600s=int((d > 600).sum()),
        absent_10min_steps=int((d[d > 600] / 600 - 1).sum()),
        longest_gap_days=round(float(d.max()) / 86400, 2), na_count=int(o.isna().sum()),
        occ_min=o.min(), occ_median=o.median(), occ_mean=round(o.mean(), 2), occ_max=o.max(),
        occ_zero_rows=int((o == 0).sum()),
    )


def reconcile_with_all_buildings(series_by_building):
    ab = pd.read_csv(C.ENERGY_DIR / "all_buildings_power.csv", na_values=["NA"])
    ab["timestamp"] = ab["timestamp"].astype("int64")
    ab = ab.set_index("timestamp")
    rows = []
    for b, s in series_by_building.items():
        col = ab[b].dropna()
        sv = s.dropna()
        common = sv.index.intersection(col.index)
        diff = (sv.loc[common] - col.loc[common]).abs()
        rows.append(dict(building=b, rows_building_file=len(s), rows_all_file_nonNA=len(col),
                         common_timestamps=len(common),
                         only_in_building_file=len(sv.index.difference(col.index)),
                         only_in_all_file=len(col.index.difference(sv.index)),
                         max_abs_diff_W=float(diff.max()), n_diff_gt_1W=int((diff > 1).sum())))
    info = dict(rows=len(ab), cols=list(ab.columns), first=str(_ist(ab.index[0])),
                last=str(_ist(ab.index[-1])), dup_ts=int(ab.index.duplicated().sum()),
                na={c: int(ab[c].isna().sum()) for c in ab.columns})
    backup = {}
    for b in C.EXCLUDED_FEEDERS:
        main = b.replace("backup", "main")
        both = ab[[main, b]].dropna()
        backup[b] = dict(rows_present=int(ab[b].notna().sum()), mean_W=float(ab[b].mean()),
                         corr_with_main=float(both.corr().iloc[0, 1]),
                         mean_ratio_backup_to_main=float(both[b].mean() / both[main].mean()))
    return pd.DataFrame(rows), info, backup, ab


def status_consistency(ab):
    st = pd.read_csv(C.ENERGY_DIR / "data_present_status_buildings.csv")
    t = pd.to_datetime(st.pop("timestamp"))
    epoch_utc = ((t - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")) - C.IST_OFFSET_S
    st.index = epoch_utc.values
    out = []
    for c in ab.columns:
        s = st[c].reindex(ab.index)
        pres = ab[c].notna()
        out.append(dict(column=c, status_rows=len(st), status_zero_total=int((st[c] == 0).sum()),
                        status_eq1_but_power_NA=int(((s == 1) & ~pres).sum()),
                        status_eq0_but_power_present=int(((s == 0) & pres).sum()),
                        grid_minutes_absent_from_all_file=int(len(st.index.difference(ab.index))),
                        share_present_pct=round(100 * st[c].mean(), 2)))
    return pd.DataFrame(out), len(st)


def audit_calendar_weather():
    cal = L.load_calendar()
    full = pd.date_range("2013-08-01", "2017-12-31")
    w = L.load_weather()
    dow = cal["date"].dt.dayofweek
    return dict(
        calendar=dict(rows=len(cal), first=str(cal.date.min().date()), last=str(cal.date.max().date()),
                      missing_dates=int(len(full.difference(pd.DatetimeIndex(cal.date)))),
                      working_day_counts=cal.working_day.value_counts().to_dict(),
                      activity_counts=cal.activity.value_counts().to_dict(),
                      weekend_marked_working=int(((dow >= 5) & (cal.working_day == 1)).sum()),
                      weekday_marked_nonworking=int(((dow < 5) & (cal.working_day == 0)).sum()),
                      crosstab_activity_workingday=pd.crosstab(cal.activity, cal.working_day).to_dict()),
        weather=dict(rows=len(w), columns=list(w.columns), first=str(w.index.min()), last=str(w.index.max()),
                     dup_index=int(w.index.duplicated().sum()),
                     na={c: int(w[c].isna().sum()) for c in w.columns},
                     overlap_with_energy_days=int(max(0, (min(w.index.max(), pd.Timestamp("2017-12-31 23:59"))
                                                          - max(w.index.min(), pd.Timestamp("2013-08-10"))).days)),
                     documented_semantics="IIITD_* = on-campus logger, Airport_* = IGIA station; "
                                          "temperature/humidity only (see I-BLEND Data Descriptor)"),
    )


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def main():
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    C.MET_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    total_grid = int((pd.Timestamp("2018-01-01") - pd.Timestamp("2013-08-10")).total_seconds() / 60)
    e_rows, series = [], {}
    for b in C.BUILDINGS:
        rec, s = audit_energy_file(b, total_grid)
        e_rows.append(rec)
        series[b] = s
        print("energy audited:", b, rec["rows"])
    e_df = pd.DataFrame(e_rows)
    e_df.to_csv(C.TAB_DIR / "audit_energy_files.csv", index=False)
    o_df = pd.DataFrame([audit_occupancy_file(b) for b in C.BUILDINGS])
    o_df.to_csv(C.TAB_DIR / "audit_occupancy_files.csv", index=False)
    rec_df, ab_info, backup, ab = reconcile_with_all_buildings(series)
    rec_df.to_csv(C.TAB_DIR / "audit_reconciliation_all_buildings.csv", index=False)
    st_df, st_rows = status_consistency(ab)
    st_df.to_csv(C.TAB_DIR / "audit_status_consistency.csv", index=False)
    m7 = list(C.BUILDINGS)
    simultaneous = float(ab[m7].notna().all(axis=1).sum()) / total_grid * 100
    cw = audit_calendar_weather()
    summary = dict(full_grid_minutes=total_grid, all_buildings=ab_info, backup_feeders=backup,
                   status_rows=st_rows, all7_present_simultaneously_pct=round(simultaneous, 2), **cw)
    with open(C.MET_DIR / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    key_e = e_df[["building", "rows", "first_ts_ist", "last_ts_ist", "dup_timestamps", "n_gaps_over_60s",
                  "absent_minutes_within_span", "longest_gap_days", "share_of_full_grid_pct",
                  "power_zero_rows"]]
    key_o = o_df[["building", "occ_code", "rows", "first_ts_ist", "last_ts_ist", "modal_step_s",
                  "absent_10min_steps", "longest_gap_days", "occ_min", "occ_max"]]
    report = ["# Dataset audit (auto-generated by src/data/audit.py)\n",
              "## Energy files\n", md_table(key_e), "\n## Occupancy files\n", md_table(key_o),
              "\n## Building files vs all_buildings_power.csv\n", md_table(rec_df),
              "\n## Status file vs power availability\n", md_table(st_df),
              "\n## Calendar / weather\n", "```json\n" + json.dumps(cw, indent=2, default=str) + "\n```\n",
              f"\nAll seven mains present in the same minute: {simultaneous:.1f}% of the 1-min grid.\n"]
    (C.REPORT_DIR / "dataset_audit.md").write_text("\n".join(report))
    print("audit complete; wrote tables, results/metrics/audit_summary.json, reports/dataset_audit.md")


if __name__ == "__main__":
    main()
