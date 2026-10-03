"""Odds ratios with bootstrap CIs and held-out permutation importance."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from .models import make_search
from .preprocessing import make_pre
from .utils import save_table


def odds_ratios(X, y, models, cfg, rng):
    """Odds ratios (per 1 SD / per category) from the LogReg model; > 1 raises risk of disease."""
    y_arr = np.asarray(y)
    n = len(y_arr)
    search = make_search(models, "LogReg", cfg).fit(X, y)
    params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    pre = make_pre().fit(X)
    Z = pre.transform(X)
    names = np.array([nm.split("__", 1)[1] for nm in pre.get_feature_names_out()])
    mk = lambda: LogisticRegression(solver="liblinear", max_iter=5000, **params)  # noqa: E731
    coef = mk().fit(Z, y_arr).coef_.ravel()
    boots = []
    for _ in range(cfg.n_boot_or):
        s = rng.integers(0, n, n)
        if len(np.unique(y_arr[s])) == 2:
            boots.append(mk().fit(Z[s], y_arr[s]).coef_.ravel())
    boots = np.array(boots)
    orr = pd.DataFrame({"feature": names, "OR": np.exp(coef),
                        "lo": np.exp(np.percentile(boots, 2.5, axis=0)),
                        "hi": np.exp(np.percentile(boots, 97.5, axis=0))})
    orr = orr[np.abs(coef) > 1e-8].sort_values("OR").reset_index(drop=True)
    orr["CI excludes 1"] = (orr.lo > 1) | (orr.hi < 1)
    save_table(orr.round(3), cfg, "odds_ratios.csv", index=False)
    return orr, params


def permutation_table(X, y, fold_res, cfg):
    """Permutation importance on each outer fold's HELD-OUT data with that fold's own tuned model."""
    imps = []
    for r in fold_res:
        te = r["te"]
        pi = permutation_importance(r["model"], X.iloc[te], y.iloc[te], scoring="roc_auc",
                                    n_repeats=3 if cfg.fast else 10, random_state=cfg.seed)
        imps.append(pi.importances_mean)
    imps = np.array(imps)
    perm = pd.DataFrame({"mean": imps.mean(0), "sd": imps.std(0)}, index=X.columns).sort_values("mean")
    save_table(perm.round(4), cfg, "permutation_importance.csv")
    return perm
