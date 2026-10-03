"""Nested calibration + operating-threshold selection, probability quality and operating points."""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .models import make_search
from .utils import boot, fmt, pick_threshold, save_table, sens_idx, sens_spec_pred, spec_idx


def _run_outer_fold(X, y, tr, te, models, name, cfg):
    """Everything here uses ONLY the training part of the fold; the test part is touched once at the end."""
    Xtr, ytr, Xte = X.iloc[tr], y.iloc[tr], X.iloc[te]
    search = make_search(models, name, cfg).fit(Xtr, ytr)
    base = search.best_estimator_                                        # tuned, refit on Xtr
    cal = CalibratedClassifierCV(clone(base), method=cfg.cal_method, cv=5).fit(Xtr, ytr)
    inner_oof = cross_val_predict(CalibratedClassifierCV(clone(base), method=cfg.cal_method, cv=5),
                                  Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=cfg.seed),
                                  method="predict_proba")[:, 1]
    return {"te": te, "model": base, "params": search.best_params_,
            "p_raw": base.predict_proba(Xte)[:, 1], "p_cal": cal.predict_proba(Xte)[:, 1],
            "thr": pick_threshold(ytr.values, inner_oof, cfg.target_sensitivity)}


def nested_calibration(X, y, models, name, cfg):
    t0 = time.time()
    folds = list(StratifiedKFold(10, shuffle=True, random_state=cfg.seed).split(X, y))
    fold_res = Parallel(n_jobs=-1)(delayed(_run_outer_fold)(X, y, tr, te, models, name, cfg) for tr, te in folds)
    print(f"Nested calibration/threshold for {name} done in {time.time() - t0:.0f}s")
    n = len(y)
    p_raw, p_cal, thr_vec = np.zeros(n), np.zeros(n), np.zeros(n)
    for r in fold_res:
        p_raw[r["te"]], p_cal[r["te"]], thr_vec[r["te"]] = r["p_raw"], r["p_cal"], r["thr"]
    thr_folds = np.array([r["thr"] for r in fold_res])
    print("Per-fold thresholds:", np.round(thr_folds, 2), "| median:", np.median(thr_folds).round(2))
    return {"fold_res": fold_res, "p_raw": p_raw, "p_cal": p_cal, "thr_vec": thr_vec,
            "thr_folds": thr_folds, "thr_final": float(np.median(thr_folds)),
            "pred_nested": p_cal >= thr_vec, "pred_05": p_cal >= 0.5}


def _cal_slope(p, y):
    q = np.clip(p, 1e-4, 1 - 1e-4)
    return LogisticRegression(C=1e6).fit(np.log(q / (1 - q)).reshape(-1, 1), y).coef_[0, 0]


def probability_quality(y_arr, cal, name, cfg):
    d = pd.DataFrame({
        "AUROC": [roc_auc_score(y_arr, cal["p_raw"]), roc_auc_score(y_arr, cal["p_cal"])],
        "Brier (lower=better)": [brier_score_loss(y_arr, cal["p_raw"]), brier_score_loss(y_arr, cal["p_cal"])],
        "Calibration slope (ideal 1)": [_cal_slope(cal["p_raw"], y_arr), _cal_slope(cal["p_cal"], y_arr)]},
        index=[f"{name} raw", f"{name} + {cfg.cal_method} calibration"]).round(3)
    save_table(d, cfg, "probability_quality.csv")
    return d


def operating_points(y_arr, cal, cfg, rng):
    """Compare fixed 0.5, naive (optimistic) and nested (honest) thresholds, with bootstrap CIs."""
    all_idx = np.arange(len(y_arr))
    naive_thr = pick_threshold(y_arr, cal["p_cal"], cfg.target_sensitivity)
    pred_naive = cal["p_cal"] >= naive_thr
    rows = []
    for label, pred in [("Calibrated @ 0.50", cal["pred_05"]),
                        (f"NAIVE thr {naive_thr:.2f} (optimistic)", pred_naive),
                        ("NESTED thr (honest)", cal["pred_nested"])]:
        s, sp = sens_idx(pred, y_arr, all_idx), spec_idx(pred, y_arr, all_idx)
        rows.append({"operating point": label,
                     "sensitivity [95% CI]": fmt(s, boot(lambda i: sens_idx(pred, y_arr, i), all_idx, rng, cfg.n_boot)),
                     "specificity [95% CI]": fmt(sp, boot(lambda i: spec_idx(pred, y_arr, i), all_idx, rng, cfg.n_boot)),
                     "precision": round(y_arr[pred].mean() if pred.any() else np.nan, 3)})
    table = pd.DataFrame(rows).set_index("operating point")
    save_table(table, cfg, "operating_points.csv")
    return table, naive_thr, pred_naive
