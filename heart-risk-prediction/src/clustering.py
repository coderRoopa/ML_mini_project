"""Unsupervised analysis done fairly: settings chosen WITHOUT labels (silhouette); ARI/NMI reported after."""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from .preprocessing import make_pre
from .utils import save_table


def _km(M, nc, k, seed):
    P = PCA(n_components=min(nc, M.shape[1]), random_state=seed).fit_transform(M)
    return P, KMeans(k, n_init=10, random_state=seed).fit_predict(P)


def run_unsupervised(X, y, df, cfg):
    seed = cfg.seed
    Z = make_pre().fit_transform(X)                          # scaled + encoded
    R = X.fillna(X.median()).values.astype(float)            # raw, unscaled (paper's setup)

    # paper's configuration (2 PCs, k=2) on raw vs scaled data
    _, lab_r = _km(R, 2, 2, seed)
    _, lab_z = _km(Z, 2, 2, seed)
    ari_paper_raw, ari_paper_scaled = adjusted_rand_score(y, lab_r), adjusted_rand_score(y, lab_z)
    print(f"Paper config (2 PCs, k=2): ARI raw = {ari_paper_raw:.3f} | ARI scaled+encoded = {ari_paper_scaled:.3f}")

    # label-free selection of (n_components, k) by silhouette
    sil = pd.DataFrame({k: {nc: silhouette_score(*_km(Z, nc, k, seed)) for nc in range(2, 11)}
                        for k in range(2, 7)})
    best_nc, best_k = [int(v) for v in sil.stack().idxmax()]
    P_sel, lab_sel = _km(Z, best_nc, best_k, seed)
    P_k2, lab_k2 = _km(Z, best_nc, 2, seed)
    print(f"Silhouette-selected: {best_nc} PCs, k = {best_k} (silhouette {sil.loc[best_nc, best_k]:.3f})")
    unsup = pd.DataFrame({
        "ARI": [adjusted_rand_score(y, lab_sel), adjusted_rand_score(y, lab_k2)],
        "NMI": [normalized_mutual_info_score(y, lab_sel), normalized_mutual_info_score(y, lab_k2)],
        "silhouette": [silhouette_score(P_sel, lab_sel), silhouette_score(P_k2, lab_k2)]},
        index=[f"selected (k={best_k})", "k=2 (matches binary outcome)"]).round(3)
    save_table(unsup, cfg, "kmeans_selected.csv")

    # information only (NOT used for selection): ARI vs number of components
    ari_raw = [adjusted_rand_score(y, _km(R, c, 2, seed)[1]) for c in range(1, 11)]
    ari_sc = [adjusted_rand_score(y, _km(Z, c, 2, seed)[1]) for c in range(1, 11)]

    # DBSCAN: eps grid from the k-distance distribution, chosen by silhouette
    MS = 10
    kd = np.sort(NearestNeighbors(n_neighbors=MS).fit(P_sel).kneighbors(P_sel)[0][:, -1])
    rows = []
    for eps in np.unique(np.round(np.quantile(kd, np.linspace(0.4, 0.98, 15)), 3)):
        lab = DBSCAN(eps=eps, min_samples=MS).fit_predict(P_sel)
        keep = lab != -1
        ncl = len(set(lab[keep]))
        rows.append({"eps": eps, "clusters": ncl, "noise %": round((~keep).mean() * 100, 1),
                     "ARI (noise as own group)": round(adjusted_rand_score(y, lab), 3),
                     "ARI (non-noise only)": round(adjusted_rand_score(y[keep], lab[keep]), 3)
                     if keep.sum() > 1 and ncl > 1 else np.nan,
                     "silhouette": round(silhouette_score(P_sel[keep], lab[keep]), 3)
                     if ncl >= 2 and keep.sum() > ncl else np.nan})
    db = pd.DataFrame(rows)
    save_table(db, cfg, "dbscan_grid.csv", index=False)
    ok = db[(db.clusters >= 2) & (db["noise %"] <= 30) & db.silhouette.notna()]
    if len(ok):
        b = ok.loc[ok.silhouette.idxmax()]
        db_msg = (f"Silhouette-selected DBSCAN: eps={b.eps}, {int(b.clusters)} clusters, "
                  f"{b['noise %']}% noise, ARI={b['ARI (noise as own group)']}")
    else:
        db_msg = "No DBSCAN setting gives >= 2 clusters with <= 30% noise: no meaningful density structure."
    print(db_msg)

    prof = df.assign(cluster=lab_k2).groupby("cluster")[
        ["age", "chol", "thalachh", "oldpeak", "caa", "sex", "exng", "risk"]].mean().round(2)
    save_table(prof, cfg, "kmeans_cluster_profiles.csv")

    return {"Z": Z, "R": R, "sil": sil, "best_nc": best_nc, "best_k": best_k, "unsup": unsup,
            "ari_paper_raw": ari_paper_raw, "ari_paper_scaled": ari_paper_scaled,
            "ari_raw": ari_raw, "ari_sc": ari_sc, "lab_k2": lab_k2,
            "P2_raw": PCA(2, random_state=seed).fit_transform(R),
            "P2_z": PCA(2, random_state=seed).fit_transform(Z),
            "evr": PCA(random_state=seed).fit(Z).explained_variance_ratio_,
            "kd": kd, "ms": MS, "db": db, "db_msg": db_msg, "prof": prof}
