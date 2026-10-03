"""All figures. Each function saves a PNG to outputs/figures and returns the Figure (shown in notebooks)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import adjusted_rand_score, brier_score_loss, roc_auc_score, roc_curve

from .utils import THS, sens_spec_pred

sns.set_theme(style="whitegrid")


def _save(fig, cfg, name):
    fig.tight_layout()
    fig.savefig(cfg.fig_dir / name, dpi=130)
    return fig


def plot_eda(df, cfg):
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    sns.heatmap(df.drop(columns="output").corr(), cmap="coolwarm", center=0, ax=ax[0], cbar=False)
    ax[0].set_title("Correlation matrix (risk = 1 means disease)")
    sns.barplot(x=df.sex.map({0: "Female", 1: "Male"}), y=df.risk, ax=ax[1], errorbar=("ci", 95))
    ax[1].set_title("High-risk rate by sex"); ax[1].set_ylabel("P(high risk)"); ax[1].set_xlabel("")
    sns.kdeplot(data=df, x="thalachh", hue="risk", fill=True, common_norm=False, ax=ax[2])
    ax[2].set_title("Max heart rate by risk")
    return _save(fig, cfg, "fig_eda.png")


def plot_split_instability(demo, cfg):
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, v in demo.items():
        sns.histplot(v, bins=20, alpha=.55, label=f"{k}: min {min(v):.2f}, max {max(v):.2f}", ax=ax)
    ax.set_xlabel("Test AUROC on one 61-patient split"); ax.legend()
    ax.set_title(f"Same model, {cfg.n_splits_demo} different random splits")
    return _save(fig, cfg, "fig_split_instability.png")


def plot_cv_boxplots(long, cfg):
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.8))
    for a, m in zip(ax, ["auroc", "recall"]):
        sns.boxplot(data=long[long.metric == m], x="model", y="value", ax=a)
        a.set_title(f"{m.upper()} across outer folds"); a.set_xlabel("")
        a.tick_params(axis="x", rotation=35)
    return _save(fig, cfg, "fig_cv_boxplots.png")


def plot_calibration_threshold(y_arr, cal, name, cfg):
    p_raw, p_cal, thr_folds, thr_final = cal["p_raw"], cal["p_cal"], cal["thr_folds"], cal["thr_final"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot([0, 1], [0, 1], "k--", label="Perfect")
    for lbl, p in [("raw", p_raw), (cfg.cal_method, p_cal)]:
        fp, mp = calibration_curve(y_arr, p, n_bins=8, strategy="quantile")
        ax[0].plot(mp, fp, marker="o", label=f"{name} {lbl} (Brier {brier_score_loss(y_arr, p):.3f})")
    ax[0].set_xlabel("Mean predicted probability"); ax[0].set_ylabel("Observed fraction high-risk")
    ax[0].set_title("Calibration (out-of-fold)"); ax[0].legend()
    fpr, tpr, _ = roc_curve(y_arr, p_cal)
    ax[1].plot(fpr, tpr, label=f"AUROC {roc_auc_score(y_arr, p_cal):.3f}"); ax[1].plot([0, 1], [0, 1], "k--")
    s_, sp_ = sens_spec_pred(y_arr, cal["pred_nested"])
    ax[1].scatter([1 - sp_], [s_], c="red", zorder=5, label="nested operating point")
    ax[1].set_xlabel("1 - specificity"); ax[1].set_ylabel("Sensitivity")
    ax[1].set_title("ROC (pooled out-of-fold)"); ax[1].legend()
    sw = pd.DataFrame([dict(thr=t, sens=sens_spec_pred(y_arr, p_cal >= t)[0],
                            spec=sens_spec_pred(y_arr, p_cal >= t)[1]) for t in THS])
    ax[2].plot(sw.thr, sw.sens, label="Sensitivity"); ax[2].plot(sw.thr, sw.spec, label="Specificity")
    ax[2].axhline(cfg.target_sensitivity, color="gray", ls=":")
    for t in thr_folds:
        ax[2].axvline(t, color="k", alpha=.15)
    ax[2].axvline(thr_final, color="k", ls="--", label=f"median nested thr = {thr_final:.2f}")
    ax[2].set_xlabel("Decision threshold"); ax[2].set_title("Trade-off (grey ticks = per-fold thresholds)")
    ax[2].legend()
    return _save(fig, cfg, "fig_calibration_threshold.png")


def plot_importance(orr, perm, name, cfg):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
    y_pos = np.arange(len(orr))
    ax[0].errorbar(orr.OR, y_pos, xerr=[orr.OR - orr.lo, orr.hi - orr.OR], fmt="o", capsize=3, color="tab:red")
    ax[0].axvline(1, color="k"); ax[0].set_xscale("log")
    ax[0].set_yticks(y_pos); ax[0].set_yticklabels(orr.feature)
    ax[0].set_title("LogReg odds ratios, 95% bootstrap CI (> 1 raises risk)"); ax[0].set_xlabel("Odds ratio (log scale)")
    ax[1].barh(perm.index, perm["mean"], xerr=perm["sd"], capsize=3)
    ax[1].set_title(f"Permutation importance ({name}, held-out folds)"); ax[1].set_xlabel("Mean drop in AUROC")
    return _save(fig, cfg, "fig_importance.png")


def plot_unsupervised(u, y, cfg):
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    sns.scatterplot(x=u["P2_raw"][:, 0], y=u["P2_raw"][:, 1], hue=y, palette="coolwarm", ax=ax[0, 0])
    ax[0, 0].set_title("PCA on RAW data (paper's setup), true risk")
    sns.scatterplot(x=u["P2_z"][:, 0], y=u["P2_z"][:, 1], hue=y, palette="coolwarm", ax=ax[0, 1])
    ax[0, 1].set_title("PCA on SCALED+encoded data, true risk")
    sns.scatterplot(x=u["P2_z"][:, 0], y=u["P2_z"][:, 1], hue=u["lab_k2"], palette="Set2", ax=ax[0, 2])
    ax[0, 2].set_title(f"k-means k=2 on {u['best_nc']} PCs (ARI {adjusted_rand_score(y, u['lab_k2']):.2f})")
    sns.heatmap(u["sil"].T, annot=True, fmt=".2f", cmap="viridis", ax=ax[1, 0])
    ax[1, 0].set_title("Silhouette: k (rows) vs # PCs (cols)")
    ax[1, 1].plot(np.cumsum(u["evr"]), marker="o"); ax[1, 1].set_xlabel("# components")
    ax[1, 1].set_title("Cumulative explained variance (scaled)")
    ax[1, 2].plot(range(1, 11), u["ari_raw"], marker="o", label="raw")
    ax[1, 2].plot(range(1, 11), u["ari_sc"], marker="o", label="scaled")
    ax[1, 2].axhline(0, color="k", ls=":"); ax[1, 2].set_xlabel("# PCs"); ax[1, 2].legend()
    ax[1, 2].set_title("[info only] ARI vs # PCs, k=2")
    return _save(fig, cfg, "fig_unsupervised.png")


def plot_dbscan(u, cfg):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(u["kd"]); ax[0].set_title(f"Sorted {u['ms']}-NN distance (k-distance plot)")
    ax[0].set_xlabel("points"); ax[0].set_ylabel("distance")
    ax[1].plot(u["db"].eps, u["db"]["ARI (noise as own group)"], marker="o")
    ax[1].set_title("DBSCAN ARI vs eps (information only)"); ax[1].set_xlabel("eps")
    return _save(fig, cfg, "fig_dbscan.png")
