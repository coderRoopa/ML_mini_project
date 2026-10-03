"""Heart-disease risk prediction: re-implementation and extension of Shi, Bechler & Ho."""
import os
import warnings

# Set at import time (before joblib starts worker processes) so workers inherit the filters.
# Only known-noisy categories are silenced: deprecation notices and optimiser convergence chatter.
_FILTERS = ("ignore:Inconsistent values:UserWarning,ignore::FutureWarning,"
            "ignore:Stochastic Optimizer:UserWarning,ignore:Training interrupted:UserWarning,"
            "ignore:The max_iter was reached:UserWarning,ignore:lbfgs failed to converge:UserWarning")
os.environ["PYTHONWARNINGS"] = ",".join(x for x in (os.environ.get("PYTHONWARNINGS", ""), _FILTERS) if x)

from sklearn.exceptions import ConvergenceWarning  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Inconsistent values")
warnings.filterwarnings("ignore", message="Training interrupted")
