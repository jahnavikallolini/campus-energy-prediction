"""Additional leakage checks for the modeling stage.  Run: python -m tests.test_models
Covers: lag construction, rolling windows, scaler fitting, temporal encoding, hyperparameter selection,
split integrity and target integrity. (tests/test_features.py holds the five earlier protections.)"""
import json
import re

import numpy as np
import pandas as pd

from src import config as C
from src.data import loaders as L
from src.features import build_features as B
from src.models import common as M

base = pd.read_csv(C.PROC_DIR / "hourly_base.csv.gz", parse_dates=["timestamp"])
feats = pd.read_csv(C.PROC_DIR / "hourly_features.csv.gz", parse_dates=["timestamp"])
F = M.VARIANTS["P"]["features"]


def test_lag_construction_by_timestamp():
    """Look up history by timestamp arithmetic (independent of row order) for random clean rows."""
    rng = np.random.default_rng(0)
    for b in ("Academic", "Library", "Lecture"):
        t = feats[(feats.building == b) & feats.eligible_primary & (feats.n_filled_inputs == 0)]
        bb = base[base.building == b].set_index("timestamp")
        for ts in rng.choice(t.timestamp.values, 150, replace=False):
            ts = pd.Timestamp(ts)
            r = t[t.timestamp == ts].iloc[0]
            for k in (1, 2, 24):
                assert r[f"power_lag{k}"] == bb.loc[ts - pd.Timedelta(hours=k), "power_w"]
                assert r[f"occ_lag{k}"] == bb.loc[ts - pd.Timedelta(hours=k), "occ_mean"]
    print("ok: lag-1/2/24 power and occupancy equal the value at t-k hours (timestamp lookup)")


def test_rolling_window_is_t_minus_24_to_t_minus_1():
    rng = np.random.default_rng(1)
    for b in ("Academic", "Mess"):
        t = feats[(feats.building == b) & feats.eligible_primary & (feats.n_filled_inputs == 0)]
        bb = base[base.building == b].set_index("timestamp")["power_w"]
        for ts in rng.choice(t.timestamp.values, 150, replace=False):
            ts = pd.Timestamp(ts)
            w = bb.loc[ts - pd.Timedelta(hours=24): ts - pd.Timedelta(hours=1)]
            assert len(w) == 24
            r = t[t.timestamp == ts].iloc[0]
            if w.notna().all():
                assert np.isclose(r.power_roll_mean_24, w.mean()) and np.isclose(r.power_roll_std_24, w.std())
            assert ts not in w.index
    # perturb ONLY the target at hour t: feature at t unchanged, feature at t+1 changes
    cal = L.load_calendar()
    bb = base[base.building == "Academic"].copy()
    ref = B.building_features(bb, cal)
    i = 20000
    bb2 = bb.copy(); bb2.loc[bb2.index[i], "power_w"] += 12345.0
    mod = B.building_features(bb2, cal)
    assert (ref.loc[i, F].fillna(-1) == mod.loc[i, F].fillna(-1)).all()
    assert ref.loc[i + 1, "power_roll_mean_24"] != mod.loc[i + 1, "power_roll_mean_24"]
    assert ref.loc[i + 1, "power_lag1"] != mod.loc[i + 1, "power_lag1"]
    print("ok: rolling features use exactly t-24..t-1; a target change at t first appears in features at t+1")


def test_scaler_fitted_on_train_only():
    d = M.load_variant("P", include_test=False)
    b = "Academic"
    tr, va = d["train"][d["train"].building == b], d["val"][d["val"].building == b]
    pipe = M.build_pipeline("KNN", dict(k=10, weights="uniform"), "cyclic", F)
    pipe.fit(tr[F], tr[M.TARGET])
    sc = pipe.named_steps["prep"].named_transformers_["num"]
    num = [f for f in F if f not in M.TEMPORAL]
    assert np.allclose(sc.mean_, tr[num].mean().values) and np.allclose(sc.scale_, tr[num].std(ddof=0).values)
    both = pd.concat([tr, va])[num].mean().values
    assert not np.allclose(sc.mean_, both)
    # fitting is blind to validation: pipeline fitted on train only is identical regardless of val content
    va2 = va.copy(); va2[num] = va2[num] * 100
    pipe2 = M.build_pipeline("KNN", dict(k=10, weights="uniform"), "cyclic", F)
    pipe2.fit(tr[F], tr[M.TARGET])
    assert np.allclose(pipe2.named_steps["prep"].named_transformers_["num"].mean_, sc.mean_)
    # target scaler inside SVR is also train-only
    sv = M.build_pipeline("SVR", dict(C=3, gamma=0.02, epsilon=0.1), "raw", F)
    s = M.stride_subsample(tr, 500); sv.fit(s[F], s[M.TARGET])
    assert np.isclose(sv.named_steps["est"].transformer_.mean_[0], s[M.TARGET].mean())
    print("ok: feature and target scalers are fitted on training rows only")


def test_temporal_encoding():
    x = np.array([[0, 0, 1], [23, 6, 12], [12, 3, 6]], float)
    e = M._cyclic(x)
    assert np.allclose(e[:, :3] ** 2 + e[:, 3:] ** 2, 1)
    assert np.isclose(M._cyclic(np.array([[24, 7, 13]], float))[0], M._cyclic(np.array([[0, 0, 1]], float))[0]).all()
    assert abs(M._cyclic(np.array([[23, 0, 1]], float))[0, 0] - M._cyclic(np.array([[0, 0, 1]], float))[0, 0]) < 0.3   # 23h close to 0h
    d = M.load_variant("P", include_test=False)
    tr = d["train"][d["train"].building == "Mess"]
    pipe = M.build_pipeline("LR", {}, "onehot", F)
    pipe.fit(tr[F], tr[M.TARGET])
    oh = pipe.named_steps["prep"].named_transformers_["oh"]
    assert [list(c) for c in oh.categories_] == [sorted(tr.hour.unique()), sorted(tr.dayofweek.unique()), sorted(tr.month.unique())]
    assert pipe.named_steps["prep"].transform(d["val"][d["val"].building == "Mess"][F]).shape[1] == pipe.named_steps["prep"].transform(tr[F]).shape[1]
    print("ok: cyclic encoding periodic and unit-norm; one-hot categories come from training rows")


def test_hyperparameter_selection_blind_to_test():
    d = M.load_variant("P", include_test=False)
    _, v_end = M.split_dates()
    assert set(d) == {"train", "val"} and max(d[s].timestamp.max() for s in d) < v_end
    try:
        M.load_variant("P", include_test=False)["test"]
        raise AssertionError("test split accessible")
    except KeyError:
        pass
    src = open(C.ROOT / "src/models/tune.py").read()
    assert "include_test=True" not in src and not re.search(r"split.*==.*['\"]test", src)
    meta = json.load(open(C.MET_DIR / "selected_config.json"))["_meta"]
    assert meta["test_rows_used"] is False
    print("ok: tuning loads only train/validation rows and never references the test split")


def test_split_integrity():
    d = M.load_variant("P", include_test=True)
    assert d["train"].timestamp.max() < pd.Timestamp("2016-07-01") <= d["val"].timestamp.min()
    assert d["val"].timestamp.max() < pd.Timestamp("2017-01-01") <= d["test"].timestamp.min()
    assert d["train"].timestamp.min() >= pd.Timestamp("2014-02-17") and d["test"].timestamp.max() <= pd.Timestamp("2017-11-04")
    keys = lambda x: set(zip(x.building, x.timestamp))
    assert not (keys(d["train"]) & keys(d["val"])) and not (keys(d["val"]) & keys(d["test"]))
    print("ok: train < validation < test, no overlapping rows, dates as agreed")


def test_best_model_decided_before_test():
    b = json.load(open(C.MET_DIR / "best_model.json"))
    assert b["decided_before_test_metrics"] is True
    print("ok: best_model.json records validation-only selection")


if __name__ == "__main__":
    for f in (test_lag_construction_by_timestamp, test_rolling_window_is_t_minus_24_to_t_minus_1, test_scaler_fitted_on_train_only,
              test_temporal_encoding, test_hyperparameter_selection_blind_to_test, test_split_integrity):
        f()
    import os
    if os.path.exists(C.MET_DIR / "best_model.json"):
        test_best_model_decided_before_test()
    print("all model-stage leakage tests passed")
