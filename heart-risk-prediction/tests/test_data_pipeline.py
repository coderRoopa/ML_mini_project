import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import Config
from src.data import load_and_clean
from src.preprocessing import make_pipe


@pytest.fixture
def csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 120
    d = rng.integers(0, 2, n)  # disease
    df = pd.DataFrame(dict(
        age=rng.normal(54, 9, n).round(), sex=rng.integers(0, 2, n), cp=rng.integers(0, 4, n),
        trtbps=rng.normal(130, 15, n).round(), chol=rng.normal(245, 40, n).round(), fbs=rng.integers(0, 2, n),
        restecg=rng.integers(0, 3, n), thalachh=(rng.normal(150, 20, n) - 20 * d).round(),
        exng=(rng.random(n) < 0.15 + 0.4 * d).astype(int), oldpeak=(rng.exponential(.7, n) + d).round(1),
        slp=rng.integers(0, 3, n), caa=np.clip(rng.poisson(.3 + d), 0, 3), thall=rng.integers(1, 4, n)))
    df.loc[0, "caa"], df.loc[1, "thall"] = 4, 0
    df["output"] = 1 - d                                     # flipped on purpose
    df = pd.concat([df, df.iloc[[5]]])                       # one duplicate
    p = tmp_path / "heart.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_cleaning_and_label_flip(csv, tmp_path):
    ds = load_and_clean(Config(data_path=csv, out_dir=str(tmp_path / "o")), verbose=False)
    assert ds.info["n_dup"] == 1 and ds.info["n_caa4"] == 1 and ds.info["n_thal0"] == 1
    assert ds.info["flipped"] is True
    assert ds.X.caa.isna().sum() == 1 and ds.X.thall.isna().sum() == 1
    assert set(ds.y.unique()) == {0, 1}


def test_pipeline_handles_missing(csv, tmp_path):
    ds = load_and_clean(Config(data_path=csv, out_dir=str(tmp_path / "o")), verbose=False)
    pipe = make_pipe(LogisticRegression(max_iter=1000)).fit(ds.X, ds.y)
    p = pipe.predict_proba(ds.X)[:, 1]
    assert p.shape == (len(ds.y),) and np.isfinite(p).all()
