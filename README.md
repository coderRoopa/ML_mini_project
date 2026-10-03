# ML_mini_project
# Heart Disease Risk Prediction: Re-implementation and Extension

Re-implementation and extension of *"Predicting myocardial infarction risk using supervised and
unsupervised machine learning methods"* (Shi, Bechler, Ho). Dataset: Kaggle
**Heart Attack Analysis & Prediction Dataset** (303 patients, 13 features).

> The outcome in this dataset is angiographic coronary artery disease (UCI Cleveland), not a documented MI.
> The notebook checks the label direction automatically (Section 2) and defines `risk = 1` as disease present.

## What is improved over the paper
| Area | Paper | This project |
|---|---|---|
| Evaluation | One 60/20/20 split (test n of about 61) | Nested, repeated stratified CV; mean ± std; corrected paired tests + one-SE model choice |
| Data quality | Not checked | Duplicates removed; invalid codes (`caa=4`, `thall=0`) imputed in-pipeline; label-direction check |
| Preprocessing | Numeric coding of nominal variables | One-hot encoding inside leakage-free `Pipeline`s |
| Models | 7 families on one split | All families (LogReg, ElasticNet, SVM linear/RBF/poly, Bagging, RF, GBM, MLP) + majority baseline |
| Calibration / threshold | None; fixed 0.5 | Calibration curves, Brier, slope, recalibration; **nested** threshold for a sensitivity target |
| Fairness | Not measured | Per-sex metrics with bootstrap CIs and a male-female gap test |
| Interpretability | None | Odds ratios with bootstrap CIs; held-out permutation importance |
| Clustering | Unscaled data; settings tuned on labels | Scaled data; silhouette-based selection; paper setup replicated side by side |
| Deliverable | None | Saved model, Streamlit demo app |

## Repository contents
```
heart_attack_risk_final.ipynb   
requirements.txt                
README.md                       
heart.csv                       
```

## Option A: run on Kaggle
1. Create a new notebook and **Upload** `heart_attack_risk_final.ipynb`.
2. **Add Input** -> search *Heart Attack Analysis & Prediction Dataset* (author: Rashik Rahman) -> Add.
3. Run All. Outputs are written to `/kaggle/working` (the *Output* tab).
4. For a ~3-minute smoke test set `FAST_MODE = True` in the config cell (Section 1). A full run takes roughly 15-30 minutes on Kaggle CPU.

## Option B: run locally
```bash
git clone <your-private-repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put heart.csv (from Kaggle) next to the notebook
jupyter notebook heart_attack_risk_final.ipynb          # then Run All
```
Outputs go to `./outputs/`.

## Live demo
After the notebook has run (so `heart_risk_model.joblib` and `app.py` exist):
```bash
cd outputs            # or /kaggle/working copy downloaded locally
pip install -r requirements.txt
streamlit run app.py
```
Enter patient values; the app shows the calibrated probability and whether it passes the screening
threshold chosen by the nested procedure. Use the same scikit-learn version that trained the model.

## Configuration (notebook Section 1)
| Setting | Default | Meaning |
|---|---|---|
| `FAST_MODE` | `False` | `True` = reduced search/bootstraps for a quick check |
| `TARGET_SENSITIVITY` | `0.95` | Clinical operating point for the nested threshold |
| `CAL_METHOD` | `"sigmoid"` | `"sigmoid"` or `"isotonic"` recalibration |
| `FLIP_LABEL` | `"auto"` | Auto-detect label direction, or force `True` / `False` |
| `SEED` | `42` | Reproducibility |

## Limitations
Single-centre, about 300 patients, no external validation; the outcome is angiographic disease rather than documented MI;
subgroup (female, n of about 96) estimates have wide confidence intervals. Not a medical device.

## Reference
Shi K., Bechler K., Ho V. *Predicting myocardial infarction risk using supervised and unsupervised machine learning methods.*
Dataset: Rahman R., *Heart Attack Analysis & Prediction Dataset*, Kaggle.
