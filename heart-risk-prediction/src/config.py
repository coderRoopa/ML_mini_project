"""Central configuration: one Config object is passed to every step."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

ROOT = Path(__file__).resolve().parents[1]

# Feature groups (dataset column names)
CAT = ["cp", "restecg", "slp", "thall"]                       # nominal -> one-hot
NUM = ["age", "trtbps", "chol", "thalachh", "oldpeak", "caa"]  # continuous / counts -> impute + scale
BIN = ["sex", "fbs", "exng"]                                  # binary -> pass-through


@dataclass
class Config:
    seed: int = 42
    fast: bool = False                       # True = reduced search/bootstraps for a quick smoke test
    target_sensitivity: float = 0.95         # clinical operating point
    cal_method: str = "sigmoid"              # "sigmoid" or "isotonic"
    flip_label: Union[str, bool] = "auto"    # "auto" = detect from data, or force True / False
    data_path: Optional[str] = None          # None = search data/, cwd, /kaggle/input
    out_dir: Optional[str] = None            # None = <repo>/outputs (or /kaggle/working/outputs)
    k_outer: int = 5

    def __post_init__(self):
        if self.out_dir is None:
            self.out_dir = ("/kaggle/working/outputs" if os.path.exists("/kaggle/working")
                            else str(ROOT / "outputs"))
        self.out = Path(self.out_dir)
        for sub in ("figures", "tables", "models"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)

    # ---- derived settings
    @property
    def n_iter(self): return 4 if self.fast else 15            # random-search iterations (inner loop)
    @property
    def mlp_n_iter(self): return min(self.n_iter, 6 if self.fast else 8)   # MLP is the slowest model
    @property
    def outer_repeats(self): return 1 if self.fast else 3
    @property
    def n_splits_demo(self): return 30 if self.fast else 200
    @property
    def n_boot(self): return 200 if self.fast else 2000
    @property
    def n_boot_or(self): return 100 if self.fast else 500
    @property
    def fig_dir(self): return self.out / "figures"
    @property
    def table_dir(self): return self.out / "tables"
    @property
    def model_dir(self): return self.out / "models"
