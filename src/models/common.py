"""Shared pieces for the regression experiments: data matrices, encoders, pipelines, metrics.

Design rules (see reports/model_evaluation_report.md):
  * One model per building, identical procedure for every building (scalers/encoders are fitted on
    that building's TRAINING rows only, inside sklearn pipelines).
  * Tuning code may only load train/validation rows (`load_variant(..., include_test=False)`).
  * Predictions are clipped at 0 W (power cannot be negative); the number of clipped values is logged.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src import config as C
from src.features.build_features import ENERGY_ONLY_FEATURES, PRIMARY_FEATURES, TARGET

TEMPORAL = ["hour", "dayofweek", "month"]
MODELS = ["LR", "KNN", "DT", "SVR"]
MODEL_NAMES = {"LR": "Linear Regression", "KNN": "KNN Regression", "DT": "Decision Tree Regression",
               "SVR": "Support Vector Regression"}
BUILDINGS = list(C.BUILDINGS)
TRAIN_START = pd.Timestamp("2014-02-17 00:00")
TEST_END = pd.Timestamp("2017-11-04 00:00")   # last hour with occupancy lag-1 available

# variant -> (feature list, which rows)
VARIANTS = {
    "P":         dict(features=PRIMARY_FEATURES,     rows="primary",      desc="energy + lagged occupancy"),
    "E_matched": dict(features=ENERGY_ONLY_FEATURES, rows="primary",      desc="energy-only, same rows as P"),
    "E_all":     dict(features=ENERGY_ONLY_FEATURES, rows="energy_only",  desc="energy-only, all eligible rows"),
}


def split_dates():
    s = json.load(open(C.MET_DIR / "split.json"))
    return pd.Timestamp(s["train_end"]), pd.Timestamp(s["val_end"])


def load_variant(variant, include_test=False):
    """Return {split: DataFrame} for a variant. Test rows are never loaded unless include_test=True."""
    spec = VARIANTS[variant]
    feats = pd.read_csv(C.PROC_DIR / "hourly_features.csv.gz", parse_dates=["timestamp"])
    t_end, v_end = split_dates()
    keep = (feats["timestamp"] >= TRAIN_START) & (feats["timestamp"] <= TEST_END)
    if not include_test:
        keep &= feats["timestamp"] < v_end          # test rows are removed before anything else happens
    flag = "eligible_primary" if spec["rows"] == "primary" else "eligible_energy_only"
    d = feats[keep & feats[flag]].copy()
    d["common"] = feats.loc[d.index, "eligible_primary"]     # row also present in the primary matrix
    d["split"] = np.where(d["timestamp"] < t_end, "train", np.where(d["timestamp"] < v_end, "val", "test"))
    if not include_test:
        assert (d["split"] != "test").all() and d["timestamp"].max() < v_end
    out = {s: d[d.split == s].reset_index(drop=True) for s in ("train", "val", "test") if include_test or s != "test"}
    return out


# ------------------------------------------------------------------ encoders
def _cyclic(X):
    X = np.asarray(X, dtype=float)                    # columns: hour, dayofweek, month
    ang = 2 * np.pi * (X - np.array([0.0, 0.0, 1.0])) / np.array([24.0, 7.0, 12.0])
    return np.hstack([np.sin(ang), np.cos(ang)])


def build_pipeline(model, params, encoding, features):
    temporal = [f for f in TEMPORAL if f in features]
    other = [f for f in features if f not in TEMPORAL]
    scale = model in ("KNN", "SVR")                   # distance/kernel methods need scaling; OLS and trees do not
    num = StandardScaler() if scale else "passthrough"
    if encoding == "raw":
        parts = [("num", num, other + temporal)]
    elif encoding == "cyclic":
        parts = [("num", num, other), ("cyc", FunctionTransformer(_cyclic), temporal)]
    elif encoding == "onehot":
        oh = OneHotEncoder(drop="first" if model == "LR" else None, sparse_output=False)
        parts = [("num", num, other), ("oh", oh, temporal)]
    else:
        raise ValueError(encoding)
    ct = ColumnTransformer(parts)
    if model == "LR":
        est = LinearRegression()
    elif model == "KNN":
        est = KNeighborsRegressor(n_neighbors=params["k"], weights=params["weights"])
    elif model == "DT":
        est = DecisionTreeRegressor(max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
                                    random_state=C.SEED)
    elif model == "SVR":
        est = TransformedTargetRegressor(
            regressor=SVR(kernel="rbf", C=params["C"], gamma=params["gamma"], epsilon=params["epsilon"]),
            transformer=StandardScaler())
    else:
        raise ValueError(model)
    return Pipeline([("prep", ct), ("est", est)])


def candidate_grid(model):
    """Candidate (encoding, params) lists. Trees use raw integers (invariant to monotone encodings)."""
    enc = ["raw", "cyclic", "onehot"]
    if model == "LR":
        return [(e, {}) for e in enc]
    if model == "KNN":
        return [(e, dict(k=k, weights=w)) for e in enc for k in (5, 10, 20, 40) for w in ("uniform", "distance")]
    if model == "DT":
        return [("raw", dict(max_depth=d, min_samples_leaf=l)) for d in (4, 6, 8, 12, None) for l in (5, 20, 50)]
    if model == "SVR":
        return [(e, dict(C=c, gamma=g, epsilon=0.1)) for e in enc for c in (1, 3, 10, 30) for g in (0.005, 0.02, 0.08)]
    raise ValueError(model)


# ------------------------------------------------------------------- metrics
def metrics(y, p, sigma_train):
    mse = mean_squared_error(y, p)
    return dict(MAE=mean_absolute_error(y, p), MSE=mse, RMSE=float(np.sqrt(mse)), R2=r2_score(y, p),
                nMAE_std=mean_absolute_error(y, p) / sigma_train, nRMSE_std=float(np.sqrt(mse)) / sigma_train,
                RMSE_pct_of_mean=100 * float(np.sqrt(mse)) / max(float(np.mean(y)), 1e-9), n=len(y))


def predict_clipped(pipe, X):
    p = pipe.predict(X)
    return np.clip(p, 0, None), int((p < 0).sum())


def fit_timed(pipe, X, y):
    t = time.perf_counter()
    pipe.fit(X, y)
    return time.perf_counter() - t


def stride_subsample(df, max_rows):
    """Deterministic chronological stride subsample (keeps all seasons represented)."""
    if len(df) <= max_rows:
        return df
    step = int(np.ceil(len(df) / max_rows))
    return df.iloc[::step]
