"""Command-line entry point:  python main.py [--fast] [--data PATH] [--out DIR]"""
import argparse

from src.config import Config
from src.pipeline import run_all


def parse_args():
    p = argparse.ArgumentParser(description="Heart-disease risk prediction: full reproducible pipeline")
    p.add_argument("--data", default=None, help="path to heart.csv (default: search data/, cwd, /kaggle/input)")
    p.add_argument("--out", default=None, help="output directory (default: ./outputs)")
    p.add_argument("--fast", action="store_true", help="quick smoke test (reduced search and bootstraps)")
    p.add_argument("--target-sensitivity", type=float, default=0.95)
    p.add_argument("--cal-method", choices=["sigmoid", "isotonic"], default="sigmoid")
    p.add_argument("--flip-label", choices=["auto", "true", "false"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    flip = {"auto": "auto", "true": True, "false": False}[a.flip_label]
    run_all(Config(seed=a.seed, fast=a.fast, target_sensitivity=a.target_sensitivity,
                   cal_method=a.cal_method, flip_label=flip, data_path=a.data, out_dir=a.out))
