"""Leakage and correctness tests for src/features/build_features.py.  Run: python -m tests.test_features"""
import numpy as np
import pandas as pd

from src import config as C
from src.data import loaders as L
from src.features import build_features as B

base = pd.read_csv(C.PROC_DIR / "hourly_base.csv.gz", parse_dates=["timestamp"])
cal = L.load_calendar()
FEATS = B.PRIMARY_FEATURES


def run(b="Academic", mutate=None):
    t = base[base.building == b].copy()
    if mutate:
        mutate(t)
    return B.building_features(t, cal)


def test_invariance_to_current_and_future():
    """Changing the target/occupancy at hour m and after must not change features of rows <= m."""
    ref = run()
    m = 20000
    def mut(t):
        idx = t.index[m:]
        t.loc[idx, "power_w"] = t.loc[idx, "power_w"] * 3 + 1000
        t.loc[idx, "occ_mean"] = t.loc[idx, "occ_mean"] * 5 + 7
    mod = run(mutate=mut)
    a, b = ref.loc[:m, FEATS], mod.loc[:m, FEATS]
    pd.testing.assert_frame_equal(a, b)
    print("ok: features of rows <= m unchanged when target/occupancy at hours >= m are altered")


def test_lag_values_match_history():
    ref = run()
    el = ref[ref.eligible_primary & (ref.n_filled_inputs == 0)].index[5000:5200]
    p = ref["power_w"]
    for i in el:
        assert ref.loc[i, "power_lag1"] == p[i - 1] and ref.loc[i, "power_lag2"] == p[i - 2]
        assert ref.loc[i, "power_lag24"] == p[i - 24]
        w = p[i - 24:i]            # hours t-24 .. t-1 (excludes t)
        assert np.isclose(ref.loc[i, "power_roll_mean_24"], w.mean())
        assert np.isclose(ref.loc[i, "power_roll_std_24"], w.std())
        o = ref["occ_mean"]
        assert ref.loc[i, "occ_lag1"] == o[i - 1] and ref.loc[i, "occ_lag24"] == o[i - 24]
    print("ok: lag/rolling values equal the raw history (rolling window excludes the current hour)")


def test_no_fill_beyond_series_end_and_target_never_filled():
    ref = run("Boys_main")
    assert (ref.loc[ref.eligible_primary, "power_w"].notna()).all()
    last_occ = ref["occ_mean"].last_valid_index()
    of = B.causal_fill(ref["occ_mean"], C.MAX_FILL_HOURS)
    assert of.loc[of.index > last_occ].isna().all()
    print("ok: targets are observed values; no fill beyond series end")


def test_calendar_is_date_level():
    ref = run()
    d = ref.groupby(ref.timestamp.dt.normalize())["working_day"].nunique()
    assert (d == 1).all()
    print("ok: calendar features are constant within a date")


def test_hour_alignment():
    """A known epoch maps to the expected IST hour: 1376073000 = 2013-08-09 18:30 UTC = 2013-08-10 00:00 IST."""
    assert L.hour_key_to_timestamp(L.hour_key([1376073000]))[0] == pd.Timestamp("2013-08-10 00:00")
    print("ok: epoch -> IST hour alignment")


if __name__ == "__main__":
    for f in (test_hour_alignment, test_invariance_to_current_and_future, test_lag_values_match_history,
              test_no_fill_beyond_series_end_and_target_never_filled, test_calendar_is_date_level):
        f()
    print("all tests passed")
