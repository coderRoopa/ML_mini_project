"""Model zoo (paper's families + majority baseline) and the inner-loop hyper-parameter search."""
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import make_pipe

# Simplest -> most complex; used by the one-standard-error rule
COMPLEXITY = ["LogReg", "ElasticNet", "SVM_Linear", "SVM_RBF", "SVM_Poly", "MLP",
              "Bagging", "RandomForest", "GradBoost"]


def build_models(cfg):
    s = cfg.seed
    return {
        "LogReg": (make_pipe(LogisticRegression(solver="liblinear", max_iter=5000)),
                   {"clf__C": loguniform(1e-3, 1e2), "clf__penalty": ["l1", "l2"]}),
        "ElasticNet": (make_pipe(LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000)),
                       {"clf__C": loguniform(1e-3, 1e2), "clf__l1_ratio": uniform(0, 1)}),
        "SVM_Linear": (make_pipe(SVC(kernel="linear", probability=True, random_state=s)),
                       {"clf__C": loguniform(1e-3, 1e1)}),
        "SVM_RBF": (make_pipe(SVC(kernel="rbf", probability=True, random_state=s)),
                    {"clf__C": loguniform(1e-1, 1e2), "clf__gamma": loguniform(1e-4, 1e-1)}),
        "SVM_Poly": (make_pipe(SVC(kernel="poly", probability=True, random_state=s)),
                     {"clf__C": loguniform(1e-2, 1e1), "clf__degree": [2, 3],
                      "clf__gamma": loguniform(1e-3, 1e-1)}),
        # early_stopping + max_iter=500 keeps the MLP fast (it was the bottleneck at max_iter=1000)
        "MLP": (make_pipe(MLPClassifier(max_iter=500, early_stopping=True, n_iter_no_change=15,
                                        validation_fraction=0.15, random_state=s)),
                {"clf__hidden_layer_sizes": [(18, 15, 10, 8), (16,), (32, 16)],  # first = paper's architecture
                 "clf__alpha": loguniform(1e-4, 1e-1), "clf__learning_rate_init": loguniform(1e-3, 3e-2)}),
        "Bagging": (make_pipe(BaggingClassifier(DecisionTreeClassifier(), random_state=s)),
                    {"clf__n_estimators": randint(10, 300), "clf__max_samples": uniform(0.5, 0.5),
                     "clf__estimator__max_depth": [None, 3, 5, 8]}),
        "RandomForest": (make_pipe(RandomForestClassifier(random_state=s, n_jobs=1)),
                         {"clf__n_estimators": randint(100, 500), "clf__max_depth": [None, 3, 5, 8],
                          "clf__min_samples_leaf": randint(1, 10), "clf__max_features": ["sqrt", "log2", 0.5]}),
        "GradBoost": (make_pipe(GradientBoostingClassifier(random_state=s)),
                      {"clf__n_estimators": randint(50, 300), "clf__learning_rate": loguniform(0.01, 0.3),
                       "clf__max_depth": randint(1, 4), "clf__subsample": uniform(0.6, 0.4)}),
    }


def make_search(models, name, cfg):
    """3-fold randomized search scored by AUROC (threshold-free)."""
    pipe, space = models[name]
    n_iter = cfg.mlp_n_iter if name == "MLP" else cfg.n_iter
    return RandomizedSearchCV(clone(pipe), space, n_iter=n_iter, scoring="roc_auc",
                              cv=StratifiedKFold(3, shuffle=True, random_state=cfg.seed),
                              random_state=cfg.seed, n_jobs=1, refit=True)
