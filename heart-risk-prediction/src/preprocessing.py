"""Leakage-safe preprocessing: lives inside every Pipeline, re-fit per training fold."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BIN, CAT, NUM


def make_pre():
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    return ColumnTransformer([("cat", cat, CAT), ("num", num, NUM), ("bin", "passthrough", BIN)],
                             sparse_threshold=0)


def make_pipe(clf):
    return Pipeline([("pre", make_pre()), ("clf", clf)])
