"""PCA of the feature space and K-Means clustering of building-day load profiles (exploratory, development window only).

Usage:  python -m src.analysis.pca_kmeans
PCA   : primary-matrix rows of the TRAINING period, features standardised per building (statistics from that building's training rows),
        pooled over buildings; PCA is used for pattern analysis only, not as a modelling step.
K-Means: one observation per building-day with 24 valid hours in 2014-02-17..2016-12-31 (train + validation, before the timestamp break),
        described by five shape features; K chosen by silhouette (elbow inertia also reported). Clusters are described by their statistics,
        not given causal labels.
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config as C
from src.features.build_features import PRIMARY_FEATURES, TARGET
from src.visualization import style

PCA_FEATURES = ["power_lag1", "power_lag2", "power_lag24", "power_roll_mean_24", "power_roll_std_24", "occ_lag1", "occ_lag2", "occ_lag24",
                "is_weekend", "working_day", "activity_high", "hour_sin", "hour_cos"]


def run_pca(feats):
    t = feats[(feats.split == "train") & feats.eligible_primary].copy()
    t["hour_sin"], t["hour_cos"] = np.sin(2 * np.pi * t.hour / 24), np.cos(2 * np.pi * t.hour / 24)
    Z = []
    for b, g in t.groupby("building"):
        sc = StandardScaler().fit(g[PCA_FEATURES])
        Z.append(pd.DataFrame(sc.transform(g[PCA_FEATURES]), columns=PCA_FEATURES, index=g.index))
    Z = pd.concat(Z).loc[t.index]
    pca = PCA(random_state=C.SEED).fit(Z)
    ev = pd.DataFrame({"component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
                       "explained_variance_ratio": pca.explained_variance_ratio_, "cumulative": np.cumsum(pca.explained_variance_ratio_)})
    ev.round(4).to_csv(C.TAB_DIR / "pca_explained_variance.csv", index=False)
    load = pd.DataFrame(pca.components_[:3].T, index=PCA_FEATURES, columns=["PC1", "PC2", "PC3"]).round(3)
    load.to_csv(C.TAB_DIR / "pca_loadings_pc1_pc3.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].bar(range(1, len(ev) + 1), ev.explained_variance_ratio * 100, color="#0072B2"); ax[0].set(xlabel="component", ylabel="explained variance (%)", title="Explained variance")
    ax[1].plot(range(1, len(ev) + 1), ev.cumulative * 100, "o-", color="#D55E00"); ax[1].axhline(90, color="k", lw=.5)
    ax[1].set(xlabel="components", ylabel="cumulative explained variance (%)", title="Cumulative explained variance")
    samp = Z.sample(min(20000, len(Z)), random_state=C.SEED); proj = pca.transform(samp)
    bl = t.loc[samp.index, "building"]
    for b in C.BUILDINGS:
        m = (bl == b).values
        ax[2].scatter(proj[m, 0], proj[m, 1], s=2, alpha=.3, color=style.BUILDING_COLORS[b], label=style.LABELS[b])
    ax[2].set(xlabel=f"PC1 ({ev.explained_variance_ratio[0] * 100:.1f}%)", ylabel=f"PC2 ({ev.explained_variance_ratio[1] * 100:.1f}%)", title="Training rows in PC1-PC2 (20,000 sampled)")
    ax[2].legend(markerscale=4, fontsize=6)
    fig.tight_layout(); style.save(fig, "fig25_pca.png")
    n90 = int((ev.cumulative < .9).sum() + 1)
    return dict(n_rows=len(Z), n_features=len(PCA_FEATURES), pc1=float(ev.explained_variance_ratio[0]), pc2=float(ev.explained_variance_ratio[1]),
                pc3=float(ev.explained_variance_ratio[2]), components_for_90pct=n90), load


def day_profiles(feats):
    d = feats[(feats.timestamp >= "2014-02-17") & (feats.timestamp < "2017-01-01") & feats[TARGET].notna()].copy()
    d["date"] = d.timestamp.dt.normalize()
    cnt = d.groupby(["building", "date"]).size()
    full = cnt[cnt == 24].index
    d = d.set_index(["building", "date"]).loc[full].reset_index()
    mu = d.groupby("building")[TARGET].transform(lambda s: s.mean())
    d["rel"] = d[TARGET] / mu
    rows = []
    for (b, dt), g in d.groupby(["building", "date"]):
        r = g.sort_values("hour").rel.values
        day, night = r[8:18].mean(), r[0:6].mean()
        rows.append(dict(building=b, date=dt, mean_rel=r.mean(), load_factor=r.mean() / max(r.max(), 1e-9), std_rel=r.std(),
                         day_night_ratio=day / max(night, 1e-3), evening_share=r[18:24].sum() / max(r.sum(), 1e-9),
                         weekend=int(dt.dayofweek >= 5), activity_high=float(g.activity_high.iloc[0]), profile=r))
    return pd.DataFrame(rows)


def run_kmeans(feats, exclude=(), tag="all7"):
    P = day_profiles(feats)
    P = P[~P.building.isin(exclude)].reset_index(drop=True)
    F = ["mean_rel", "load_factor", "std_rel", "day_night_ratio", "evening_share"]
    P["day_night_ratio"] = np.log(P.day_night_ratio.clip(0.05, 20))
    X = StandardScaler().fit_transform(P[F])
    res = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=C.SEED).fit(X)
        res.append(dict(k=k, inertia=km.inertia_, silhouette=silhouette_score(X, km.labels_, sample_size=6000, random_state=C.SEED)))
    sel = pd.DataFrame(res); sel.round(4).to_csv(C.TAB_DIR / f"kmeans_{tag}_k_selection.csv", index=False)
    K = int(sel.loc[sel.silhouette.idxmax(), "k"])
    km = KMeans(n_clusters=K, n_init=10, random_state=C.SEED).fit(X)
    P["cluster"] = km.labels_
    summ = P.groupby("cluster").agg(days=("date", "size"), mean_rel=("mean_rel", "mean"), load_factor=("load_factor", "mean"), std_rel=("std_rel", "mean"),
                                    day_night_ratio_log=("day_night_ratio", "mean"), evening_share=("evening_share", "mean"),
                                    weekend_share=("weekend", "mean"), high_activity_share=("activity_high", "mean")).round(3)
    summ.to_csv(C.TAB_DIR / f"kmeans_{tag}_cluster_summary.csv")
    comp = pd.crosstab(P.building, P.cluster); comp.to_csv(C.TAB_DIR / f"kmeans_{tag}_building_by_cluster_days.csv")
    P.drop(columns="profile").to_csv(C.TAB_DIR / f"kmeans_{tag}_building_day_assignments.csv.gz", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax[0].plot(sel.k, sel.inertia, "o-", color="#0072B2"); ax[0].set(xlabel="K", ylabel="inertia", title="Elbow")
    ax[1].plot(sel.k, sel.silhouette, "o-", color="#D55E00"); ax[1].axvline(K, color="k", lw=.5, ls="--"); ax[1].set(xlabel="K", ylabel="silhouette", title=f"Silhouette (K = {K}, {tag})")
    cols = plt.cm.tab10(np.arange(K))
    for c in range(K):
        prof = np.vstack(P[P.cluster == c].profile.values).mean(axis=0)
        ax[2].plot(range(24), prof, color=cols[c], label=f"cluster {c} (n={int((P.cluster == c).sum())})")
    ax[2].set(xlabel="hour of day", ylabel="power / building mean", title="Mean normalised daily profile per cluster"); ax[2].legend(fontsize=7)
    fig.tight_layout(); style.save(fig, f"fig26_kmeans_{tag}.png")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    share = comp.div(comp.sum(axis=1), axis=0) * 100
    im = ax.imshow(share.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_yticks(range(len(share))); ax.set_yticklabels([style.LABELS[b] for b in share.index]); ax.set_xticks(range(K)); ax.set_xticklabels([f"cluster {c}" for c in range(K)]); ax.grid(False)
    for i in range(share.shape[0]):
        for j in range(K):
            ax.text(j, i, f"{share.values[i, j]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="% of the building's days"); ax.set_title("Building-days by cluster (%)")
    style.save(fig, f"fig27_kmeans_membership_{tag}.png")
    return dict(n_building_days=len(P), selected_K=K, silhouette_at_K=float(sel.loc[sel.k == K, "silhouette"].iloc[0])), summ, comp


def main():
    style.apply()
    feats = pd.read_csv(C.PROC_DIR / "hourly_features.csv.gz", parse_dates=["timestamp"])
    pm, load = run_pca(feats)
    km, summ, comp = run_kmeans(feats)
    km2, summ2, comp2 = run_kmeans(feats, exclude=("Lecture",), tag="without_lecture")
    json.dump(dict(pca=pm, kmeans_all7=km, kmeans_without_lecture=km2), open(C.MET_DIR / "pca_kmeans_summary.json", "w"), indent=2)
    pd.set_option("display.width", 200)
    print(pm); print(load); print(km); print(summ.to_string()); print(comp.to_string()); print(km2); print(summ2.to_string()); print(comp2.to_string())


if __name__ == "__main__":
    main()
