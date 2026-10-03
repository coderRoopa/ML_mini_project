"""Small metric / bootstrap helpers shared across modules."""
import numpy as np
from sklearn.metrics import roc_auc_score

THS = np.linspace(0.05, 0.95, 91)   # candidate decision thresholds


def sens_spec_pred(yt, pred):
    """Sensitivity and specificity of boolean predictions `pred` against labels `yt`."""
    yt, pred = np.asarray(yt), np.asarray(pred, dtype=bool)
    pos, neg = yt == 1, yt == 0
    return (pred[pos].mean() if pos.any() else np.nan), ((~pred[neg]).mean() if neg.any() else np.nan)


def pick_threshold(yt, pp, target, ths=THS):
    """Highest threshold whose sensitivity still reaches `target` (falls back to the lowest threshold)."""
    ok = [t for t in ths if sens_spec_pred(yt, np.asarray(pp) >= t)[0] >= target]
    return float(max(ok)) if ok else float(ths.min())


def sens_idx(pred, y, idx):
    pos = y[idx] == 1
    return pred[idx][pos].mean() if pos.any() else np.nan


def spec_idx(pred, y, idx):
    neg = y[idx] == 0
    return (~pred[idx][neg]).mean() if neg.any() else np.nan


def auc_idx(p, y, idx):
    yt = y[idx]
    return roc_auc_score(yt, p[idx]) if len(np.unique(yt)) == 2 else np.nan


def boot(fn, idx, rng, B):
    """Percentile 95% CI of fn(resampled_idx) over B patient-level bootstrap resamples."""
    vals = np.array([fn(rng.choice(idx, len(idx))) for _ in range(B)])
    vals = vals[np.isfinite(vals)]
    return np.percentile(vals, [2.5, 97.5])


def fmt(v, ci):
    return f"{v:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def save_table(df, cfg, name, index=True):
    df.to_csv(cfg.table_dir / name, index=index)
