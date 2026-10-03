"""Loading, cleaning and label-direction checking."""
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT

# Expected sign of corr(label, marker) IF label == 1 means DISEASE
EXPECT = {"exng": +1, "oldpeak": +1, "caa": +1, "thalachh": -1}


@dataclass
class Dataset:
    df: pd.DataFrame      # cleaned frame incl. `output` (original) and `risk` (1 = disease)
    X: pd.DataFrame
    y: pd.Series
    info: dict


def find_csv(path=None):
    if path:
        if not Path(path).exists():
            raise FileNotFoundError(path)
        return str(path)
    cands = sorted(glob.glob(str(ROOT / "data" / "*.csv")))
    cands += ["heart.csv"] if Path("heart.csv").exists() else []
    cands += [p for p in glob.glob("/kaggle/input/**/*.csv", recursive=True)
              if "heart" in Path(p).name.lower()]
    for p in cands:
        if {"age", "sex", "cp", "chol"} <= set(pd.read_csv(p, nrows=1).columns):
            return p
    raise FileNotFoundError(
        "heart.csv not found. Download 'Heart Attack Analysis & Prediction Dataset' from Kaggle and put it in "
        "data/ (see data/README.md), or pass --data PATH. On Kaggle: Add Input -> the dataset.")


def load_and_clean(cfg, verbose=True):
    path = find_csv(cfg.data_path)
    df = pd.read_csv(path).rename(columns={"trestbps": "trtbps", "thalach": "thalachh", "exang": "exng",
                                           "slope": "slp", "ca": "caa", "thal": "thall",
                                           "target": "output", "risk": "output"})
    info = {"path": path, "raw_shape": df.shape}

    # 1) duplicates
    info["n_dup"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # 2) invalid codes -> NaN (imputed inside the pipeline, so no leakage)
    info["n_caa4"], info["n_thal0"] = int((df.caa == 4).sum()), int((df.thall == 0).sum())
    df.loc[df.caa == 4, "caa"] = np.nan
    df.loc[df.thall == 0, "thall"] = np.nan

    # 3) label-direction check
    corr = {m: df[m].corr(df["output"]) for m in EXPECT}
    info["agree"] = int(sum(np.sign(corr[m]) == EXPECT[m] for m in EXPECT))
    info["evidence"] = pd.DataFrame({"corr(output, marker)": {m: round(c, 3) for m, c in corr.items()},
                                     "expected if output=1 is disease": EXPECT})
    if cfg.flip_label == "auto":
        flip = info["agree"] <= 1
        info["ambiguous"] = info["agree"] == 2
    else:
        flip, info["ambiguous"] = bool(cfg.flip_label), False
    info["flipped"] = bool(flip)
    df["risk"] = 1 - df["output"] if flip else df["output"]

    y = df["risk"].astype(int)
    X = df.drop(columns=["output", "risk"])
    if verbose:
        print("Reading:", path, "| raw shape:", info["raw_shape"])
        print(f"Duplicates removed: {info['n_dup']} | caa=4 -> NaN: {info['n_caa4']} | thall=0 -> NaN: {info['n_thal0']}")
        print(info["evidence"].to_string())
        print(f"{info['agree']}/4 markers agree with 'output=1 means disease'.")
        if info["ambiguous"]:
            print("WARNING: label direction ambiguous; not flipping. Set flip_label manually.")
        print("=> Label FLIPPED: risk = 1 - output" if flip else "=> Label kept: risk = output")
        print(f"Final shape {X.shape} | high-risk prevalence {y.mean():.3f} | "
              f"women {(X.sex == 0).sum()} / men {(X.sex == 1).sum()}")
    return Dataset(df=df, X=X, y=y, info=info)
