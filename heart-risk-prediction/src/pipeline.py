"""End-to-end orchestration. `run_all(cfg)` reproduces every table/figure in outputs/."""
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import adjusted_rand_score

from . import plots
from .calibration import nested_calibration, operating_points, probability_quality
from .clustering import run_unsupervised
from .data import load_and_clean
from .evaluation import run_cv, select_model, split_instability, summarize_cv
from .fairness import sex_analysis
from .interpret import odds_ratios, permutation_table
from .models import build_models, make_search
from .utils import sens_spec_pred


def export_model(X, y, models, name, cal, flipped, cfg):
    """Fit the chosen model on all data, recalibrate, and save it with its screening threshold."""
    full = make_search(models, name, cfg).fit(X, y)
    final_model = CalibratedClassifierCV(clone(full.best_estimator_), method=cfg.cal_method, cv=5).fit(X, y)
    artifact = {"model": final_model, "threshold": cal["thr_final"], "features": list(X.columns),
                "model_name": name, "label_flipped": bool(flipped), "sklearn_version": sklearn.__version__,
                "label_meaning": "1 = heart disease present / high risk"}
    path = cfg.model_dir / "heart_risk_model.joblib"
    joblib.dump(artifact, path)
    print("Saved model:", path, "| threshold:", round(cal["thr_final"], 3))
    return artifact


def build_summary(R):
    cfg, sel, summ = R["cfg"], R["sel"], R["summary"]
    fn, auc, y_arr, cal = sel["final"], sel["auc"], R["y_arr"], R["cal"]
    s_n, sp_n = sens_spec_pred(y_arr, cal["pred_nested"])
    s_v, sp_v = sens_spec_pred(y_arr, R["pred_naive"])
    s05, sp05 = sens_spec_pred(y_arr, cal["pred_05"])
    sig = sel["comp"][sel["comp"]["p (Holm)"] < 0.05].index.tolist()
    fg, u, pq = R["gap_df"].loc["Sensitivity"], R["unsup"], R["prob_q"]
    demo = R["demo"]
    return f"""
PAPER (single 60/20/20 split, test n ~ 61): LogReg AUROC 0.86-0.87, recall 0.94; SVM-RBF collapsed (AUROC 0.50); clustering ARI 0.02-0.06.

THIS WORK (nested repeated CV, {cfg.k_outer * cfg.outer_repeats} outer folds, mean ± std; label {'FLIPPED (risk = 1 - output)' if R['info']['flipped'] else 'as provided'}):
  Best by mean AUROC: {sel['best']} ({auc[sel['best']].mean():.3f}); simplest within 1 corrected SE: {fn} ({auc[fn].mean():.3f})
  Models significantly worse than the best (Holm p < 0.05): {sig if sig else 'none -> model differences are within noise'}
  Final model {fn}: AUROC {summ.loc[fn, 'auroc']} | recall {summ.loc[fn, 'recall']} | F1 {summ.loc[fn, 'f1']}
  Majority baseline accuracy: {summ.loc['Majority baseline', 'accuracy']}; SVM-RBF AUROC (pipelined+tuned): {summ.loc['SVM_RBF', 'auroc']} (paper: 0.50)
  Single-split LogReg AUROC ranged {min(demo['LogReg']):.2f}-{max(demo['LogReg']):.2f} over {cfg.n_splits_demo} seeds -> one split is unreliable.
  Calibration: Brier raw {pq.iloc[0, 1]:.3f} -> recalibrated {pq.iloc[1, 1]:.3f}; slope {pq.iloc[0, 2]:.2f} -> {pq.iloc[1, 2]:.2f}.
  Operating point (target sensitivity {cfg.target_sensitivity}):
      @0.50          : sens {s05:.3f} / spec {sp05:.3f}
      naive thr {R['naive_thr']:.2f}  : sens {s_v:.3f} / spec {sp_v:.3f}   (optimistic)
      nested thr (median {cal['thr_final']:.2f}): sens {s_n:.3f} / spec {sp_n:.3f}   (honest)
  Sex gap in sensitivity (male - female): {fg['male - female']} {fg['95% CI']}; CI excludes 0: {fg['CI excludes 0?']}
  Clustering: paper config ARI raw {u['ari_paper_raw']:.2f} vs scaled {u['ari_paper_scaled']:.2f}; silhouette-selected ({u['best_nc']} PCs, k={u['best_k']}) ARI {u['unsup'].iloc[0, 0]:.2f}.
"""


def run_all(cfg):
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    R = {"cfg": cfg}

    print("\n=== 1. Data ===")
    ds = load_and_clean(cfg)
    X, y = ds.X, ds.y
    y_arr = y.values
    R.update(info=ds.info, X=X, y=y, y_arr=y_arr)
    plots.plot_eda(ds.df, cfg)

    print("\n=== 2. Single-split instability ===")
    R["demo"] = split_instability(X, y, cfg)
    plots.plot_split_instability(R["demo"], cfg)

    print("\n=== 3. Nested CV, all models ===")
    models = build_models(cfg)
    R["models"] = models
    cv = run_cv(models, X, y, cfg)
    R["summary"], long = summarize_cv(cv, cfg)
    print(R["summary"].to_string())
    plots.plot_cv_boxplots(long, cfg)

    print("\n=== 4. Model selection ===")
    R["sel"] = select_model(cv, models, cfg)
    print(R["sel"]["comp"].to_string())
    name = R["sel"]["final"]

    print("\n=== 5. Calibration + nested threshold ===")
    cal = nested_calibration(X, y, models, name, cfg)
    R["cal"] = cal
    R["prob_q"] = probability_quality(y_arr, cal, name, cfg)
    print(R["prob_q"].to_string())
    op, R["naive_thr"], R["pred_naive"] = operating_points(y_arr, cal, cfg, rng)
    print(op.to_string())
    plots.plot_calibration_threshold(y_arr, cal, name, cfg)

    print("\n=== 6. Performance by sex ===")
    R["by_sex"], R["gap_df"] = sex_analysis(X, y_arr, cal, cfg, rng)
    print(R["by_sex"].to_string()); print(R["gap_df"].to_string())

    print("\n=== 7. Interpretability ===")
    orr, _ = odds_ratios(X, y, models, cfg, rng)
    perm = permutation_table(X, y, cal["fold_res"], cfg)
    print(orr.round(3).to_string())
    plots.plot_importance(orr, perm, name, cfg)

    print("\n=== 8. Unsupervised ===")
    R["unsup"] = run_unsupervised(X, y, ds.df, cfg)
    plots.plot_unsupervised(R["unsup"], y, cfg)
    plots.plot_dbscan(R["unsup"], cfg)

    print("\n=== 9. Export ===")
    R["artifact"] = export_model(X, y, models, name, cal, ds.info["flipped"], cfg)

    txt = build_summary(R)
    (cfg.out / "final_summary.txt").write_text(txt)
    print(txt)
    print(f"Total time: {(time.time() - t0) / 60:.1f} min | outputs in {cfg.out}")
    return R
