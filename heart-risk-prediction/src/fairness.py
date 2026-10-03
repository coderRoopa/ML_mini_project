"""Performance by sex with bootstrap CIs and male-minus-female gap tests."""
import numpy as np
import pandas as pd

from .utils import auc_idx, boot, fmt, save_table, sens_idx, spec_idx


def sex_analysis(X, y_arr, cal, cfg, rng):
    p_cal, pred_nested, pred_05 = cal["p_cal"], cal["pred_nested"], cal["pred_05"]
    sex = X["sex"].values
    groups = {"Female": np.where(sex == 0)[0], "Male": np.where(sex == 1)[0], "All": np.arange(len(y_arr))}
    B = cfg.n_boot
    rows = []
    for g, idx in groups.items():
        rows.append({"group": g, "n": len(idx), "high-risk rate": round(y_arr[idx].mean(), 2),
                     "AUROC [95% CI]": fmt(auc_idx(p_cal, y_arr, idx), boot(lambda i: auc_idx(p_cal, y_arr, i), idx, rng, B)),
                     "Sensitivity [95% CI]": fmt(sens_idx(pred_nested, y_arr, idx), boot(lambda i: sens_idx(pred_nested, y_arr, i), idx, rng, B)),
                     "Specificity [95% CI]": fmt(spec_idx(pred_nested, y_arr, idx), boot(lambda i: spec_idx(pred_nested, y_arr, i), idx, rng, B)),
                     "Sensitivity @0.5": round(sens_idx(pred_05, y_arr, idx), 3)})
    by_sex = pd.DataFrame(rows).set_index("group")
    save_table(by_sex, cfg, "performance_by_sex.csv")

    def gap(fn):
        f, m = groups["Female"], groups["Male"]
        vals = [fn(rng.choice(m, len(m))) - fn(rng.choice(f, len(f))) for _ in range(B)]
        vals = np.array(vals)[np.isfinite(vals)]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return fn(m) - fn(f), lo, hi, not (lo <= 0 <= hi)

    rows = []
    for nm, fn in [("AUROC", lambda i: auc_idx(p_cal, y_arr, i)),
                   ("Sensitivity", lambda i: sens_idx(pred_nested, y_arr, i)),
                   ("Specificity", lambda i: spec_idx(pred_nested, y_arr, i))]:
        o, lo, hi, sig = gap(fn)
        rows.append({"metric": nm, "male - female": round(o, 3), "95% CI": f"[{lo:.2f}, {hi:.2f}]",
                     "CI excludes 0?": "YES" if sig else "no"})
    gap_df = pd.DataFrame(rows).set_index("metric")
    save_table(gap_df, cfg, "sex_gaps.csv")
    return by_sex, gap_df
