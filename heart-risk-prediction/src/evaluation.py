"""Nested repeated CV for all models, corrected paired tests and one-SE model selection."""
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split

from .models import COMPLEXITY, make_search
from .preprocessing import make_pipe
from .utils import save_table

METRICS = ["auroc", "accuracy", "precision", "recall", "f1", "brier"]


# All threshold-based metrics use predict_proba >= 0.5, NOT predict(): for SVC, predict() (decision
# function) and predict_proba() (Platt scaling) can disagree and make an SVM look "collapsed" at 0.5.
def _thr_scorer(fn):
    return lambda est, Xt, yt: fn(yt, (est.predict_proba(Xt)[:, 1] >= 0.5).astype(int))


SCORING = {
    "auroc": lambda est, Xt, yt: roc_auc_score(yt, est.predict_proba(Xt)[:, 1]),
    "accuracy": _thr_scorer(accuracy_score),
    "precision": _thr_scorer(lambda a, b: precision_score(a, b, zero_division=0)),
    "recall": _thr_scorer(lambda a, b: recall_score(a, b, zero_division=0)),
    "f1": _thr_scorer(lambda a, b: f1_score(a, b, zero_division=0)),
    "brier": "neg_brier_score",
}


def split_instability(X, y, cfg):
    """Repeat the paper's single 60/20/20 protocol with many seeds to show how unstable it is."""
    def one(seed, clf):
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed)
        _, X_te, _, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed)
        return roc_auc_score(y_te, make_pipe(clf).fit(X_tr, y_tr).predict_proba(X_te)[:, 1])
    seeds = range(cfg.n_splits_demo)
    return {"LogReg": [one(s, LogisticRegression(max_iter=5000)) for s in seeds],
            "RandomForest": [one(s, RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1))
                             for s in seeds]}


def run_cv(models, X, y, cfg, verbose=True):
    """Nested CV: inner RandomizedSearch (AUROC) inside an outer repeated stratified K-fold."""
    outer = RepeatedStratifiedKFold(n_splits=cfg.k_outer, n_repeats=cfg.outer_repeats, random_state=cfg.seed)
    results = {"Majority baseline": cross_validate(DummyClassifier(strategy="most_frequent"), X, y,
                                                   cv=outer, scoring=SCORING)}
    for name in models:
        t0 = time.time()
        results[name] = cross_validate(make_search(models, name, cfg), X, y, cv=outer,
                                       scoring=SCORING, n_jobs=-1)
        if verbose:
            print(f"{name:13s} done in {time.time() - t0:5.0f}s", flush=True)
    return results


def summarize_cv(results, cfg):
    rows, long = [], []
    for name, res in results.items():
        row = {"model": name}
        for m in METRICS:
            v = res["test_" + m] * (-1 if m == "brier" else 1)
            row[m] = f"{v.mean():.3f} ± {v.std():.3f}"
            long += [{"model": name, "metric": m, "value": x} for x in v]
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("model")
    save_table(summary, cfg, "cv_summary.csv")
    return summary, pd.DataFrame(long)


def corrected_se(d, k):
    """Nadeau-Bengio corrected standard error of the mean of fold-wise differences."""
    n = len(d)
    return np.sqrt((1 / n + 1 / (k - 1)) * np.var(d, ddof=1))


def nb_test(a, b, k):
    d = a - b
    se = corrected_se(d, k)
    t = d.mean() / se if se > 0 else 0.0
    return d.mean(), 2 * stats.t.sf(abs(t), len(d) - 1)


def select_model(results, models, cfg):
    """Corrected paired t-tests vs the best model (Holm-adjusted) + one-standard-error rule."""
    k = cfg.k_outer
    auc = {n: r["test_auroc"] for n, r in results.items() if n in models}
    best = max(auc, key=lambda n: auc[n].mean())
    comp = pd.DataFrame([{"model": n, "mean AUROC": auc[n].mean(),
                          "diff vs best": nb_test(auc[n], auc[best], k)[0],
                          "p (corrected t)": nb_test(auc[n], auc[best], k)[1]}
                         for n in auc if n != best]).sort_values("p (corrected t)")
    m = len(comp)
    comp["p (Holm)"] = np.maximum.accumulate((m - np.arange(m)) * comp["p (corrected t)"].values).clip(max=1.0)
    comp = comp.set_index("model").round(4)
    save_table(comp, cfg, "model_comparison_tests.csv")
    se_best = corrected_se(auc[best], k) if len(auc[best]) > 1 else 0.0
    candidates = [n for n in COMPLEXITY if n in auc and auc[n].mean() >= auc[best].mean() - se_best]
    final = candidates[0]
    print(f"Best by mean AUROC : {best} ({auc[best].mean():.3f})")
    print(f"Within 1 corrected SE ({se_best:.3f}) of best: {candidates}")
    print(f"=> FINAL MODEL (simplest within 1 SE): {final} ({auc[final].mean():.3f})")
    return {"auc": auc, "best": best, "final": final, "candidates": candidates,
            "se_best": se_best, "comp": comp}
