"""Score new patients with the saved model:  python predict.py --input patients.csv

The CSV needs the 13 feature columns (age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng,
oldpeak, slp, caa, thall). Output: probability of heart disease and a flag at the screening threshold.
"""
import argparse

import joblib
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--model", default="outputs/models/heart_risk_model.joblib")
p.add_argument("--output", default=None, help="optional CSV path for the scored table")
a = p.parse_args()

art = joblib.load(a.model)
df = pd.read_csv(a.input)
df = df.rename(columns={"trestbps": "trtbps", "thalach": "thalachh", "exang": "exng",
                        "slope": "slp", "ca": "caa", "thal": "thall"})
df.loc[df.get("caa", pd.Series(dtype=float)) == 4, "caa"] = float("nan")      # invalid codes -> missing,
df.loc[df.get("thall", pd.Series(dtype=float)) == 0, "thall"] = float("nan")  # same as in training
missing = set(art["features"]) - set(df.columns)
if missing:
    raise SystemExit(f"Missing columns: {sorted(missing)}")
prob = art["model"].predict_proba(df[art["features"]])[:, 1]
out = df.assign(probability=prob.round(3), flagged_high_risk=prob >= art["threshold"])
print(f"Model: {art['model_name']} | screening threshold: {art['threshold']:.2f} | positive class: {art['label_meaning']}")
print(out.to_string(index=False))
if a.output:
    out.to_csv(a.output, index=False)
