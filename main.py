import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fredapi import Fred
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from dotenv import load_dotenv
load_dotenv()
from sklearn.model_selection import TimeSeriesSplit
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    raise EnvironmentError(
        "FRED_API_KEY not set.\n"
        "  export FRED_API_KEY=your_key_here"
    )

fred = Fred(api_key=FRED_API_KEY)

series = {
    "UNRATE":    "UNRATE",
    "CPI":       "CPIAUCSL",
    "10Y":       "DGS10",
    "2Y":        "DGS2",
    "CONF":      "UMCSENT",
    "INDPRO":    "INDPRO",
    "RECESSION": "USREC",
}

print("Fetching data from FRED...")
df_raw = pd.DataFrame()
for name, code in series.items():
    s = fred.get_series(code)
    if s is None or s.empty:
        raise ValueError(f"Could not fetch series '{code}' from FRED.")
    df_raw[name] = s

data = df_raw.resample("ME").last().ffill()
print(f"Data loaded: {data.index[0].date()} → {data.index[-1].date()} "
      f"({len(data)} months, {int(data['RECESSION'].sum())} recession months)")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["yield_curve"]   = df["10Y"] - df["2Y"]
    df["unrate_ma3"]    = df["UNRATE"].rolling(3).mean()
    df["unrate_min_12"] = df["UNRATE"].rolling(12).min()
    df["sahm_rule"]     = df["unrate_ma3"] - df["unrate_min_12"]
    df["cpi_yoy"]       = df["CPI"].pct_change(12)
    df["indpro_mom"]    = df["INDPRO"].pct_change(1)
    for col in ["yield_curve", "sahm_rule", "CONF", "UNRATE", "cpi_yoy"]:
        for lag in [1, 3, 6, 12]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    # Target: any recession month in the next 6 months
    df["target"] = (
        df["RECESSION"]
        .rolling(6)
        .max()
        .shift(-6)
    )
    return df


data_full = engineer_features(data)
FEATURE_COLS = [
    c for c in data_full.columns
    if c not in ("RECESSION", "target")
]

data_clean = data_full.dropna(subset=FEATURE_COLS + ["target"])
print(f"Clean rows: {len(data_clean)} | recession-positive months: {int(data_clean['target'].sum())}")

N_SPLITS = 8
tscv     = TimeSeriesSplit(n_splits=N_SPLITS, gap=6)

X_all = data_clean[FEATURE_COLS]
y_all = data_clean["target"]

def tune_fold(X_tr: np.ndarray, y_tr: np.ndarray, n_trials: int = 40) -> dict:
    ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    inner = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 500),
            max_depth        = trial.suggest_int("max_depth", 2, 5),
            learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            scale_pos_weight = ratio,
            random_state     = 42,
            eval_metric      = "logloss",
            verbosity        = 0,
        )
        scores = []
        for itr, ival in inner.split(X_tr):
            if y_tr[ival].sum() == 0:
                continue
            m = XGBClassifier(**params)
            m.fit(X_tr[itr], y_tr[itr])
            p = m.predict_proba(X_tr[ival])[:, 1]
            scores.append(average_precision_score(y_tr[ival], p))
        return np.mean(scores) if scores else 0.0

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

oos_probs  = np.full(len(data_clean), np.nan)
oos_preds  = np.full(len(data_clean), np.nan)

X_np = X_all.values
y_np = y_all.values

for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_all), 1):
    print(f"\nFold {fold}/{N_SPLITS}  |  train={len(tr_idx)}  test={len(te_idx)}  "
          f"positives_in_test={int(y_np[te_idx].sum())}")

    X_tr, y_tr = X_np[tr_idx], y_np[tr_idx]
    X_te, y_te = X_np[te_idx], y_np[te_idx]

    if y_tr.sum() == 0:
        print("  Skipping fold — no positives in training set.")
        continue

    best_params = tune_fold(X_tr, y_tr, n_trials=40)
    ratio       = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    base = XGBClassifier(
        **best_params,
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    if y_tr.sum() >= 10:
        fold_model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    else:
        fold_model = CalibratedClassifierCV(base, method="sigmoid", cv=3)

    fold_model.fit(X_tr, y_tr)

    probs_te = fold_model.predict_proba(X_te)[:, 1]
    preds_te = (probs_te >= 0.4).astype(float)

    oos_probs[te_idx] = probs_te
    oos_preds[te_idx] = preds_te

    pr_auc = average_precision_score(y_te, probs_te) if y_te.sum() > 0 else float("nan")
    print(f"  Fold PR-AUC: {pr_auc:.4f}")

mask       = ~np.isnan(oos_probs)
eval_index = data_clean.index[mask]
y_eval     = y_np[mask]
p_eval     = oos_probs[mask]
d_eval     = oos_preds[mask]

print("\n\n--- WALK-FORWARD OOS CLASSIFICATION REPORT ---")
print(classification_report(y_eval, d_eval, digits=3))

pr_auc  = average_precision_score(y_eval, p_eval)
roc_auc = roc_auc_score(y_eval, p_eval)
print(f"Precision-Recall AUC : {pr_auc:.4f}  ← primary metric")
print(f"ROC-AUC              : {roc_auc:.4f}")

print("\nTraining final model on full dataset...")
ratio_full  = (y_np == 0).sum() / max((y_np == 1).sum(), 1)
best_global = tune_fold(X_np, y_np, n_trials=50)
final_base = XGBClassifier(
    **best_global,
    scale_pos_weight=ratio_full,
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)
final_model = CalibratedClassifierCV(final_base, method="isotonic", cv=5)
final_model.fit(X_np, y_np)
imp_arrays = [e.estimator.feature_importances_ for e in final_model.calibrated_classifiers_]
mean_imp   = pd.Series(np.mean(imp_arrays, axis=0), index=FEATURE_COLS)
print("\n--- TOP 10 PREDICTORS ---")
print(mean_imp.sort_values(ascending=False).head(10).to_string())
latest_row   = data_full[FEATURE_COLS].dropna().iloc[[-1]]
current_risk = final_model.predict_proba(latest_row)[0][1]

rng        = np.random.default_rng(42)
boot_means = [
    p_eval[rng.integers(0, len(p_eval), len(p_eval))].mean()
    for _ in range(2000)
]
scale    = current_risk / (p_eval.mean() + 1e-9)
ci_lo    = np.percentile(boot_means, 2.5)  * scale
ci_hi    = np.percentile(boot_means, 97.5) * scale
print(f"\n{'='*52}")
print(f"LATEST UPDATE  ({latest_row.index[0].date()})")
print(f"Recession probability (next 6 months): {current_risk:.1%}")
print(f"95 % bootstrap CI: [{min(ci_lo,1):.1%}, {min(ci_hi,1):.1%}]")
print(f"{'='*52}")

rec_periods = data_clean["RECESSION"][mask]

fig, axes = plt.subplots(3, 1, figsize=(14, 13))

# OOS predicted probability over time
ax0 = axes[0]
ax0.plot(eval_index, p_eval, label="OOS Predicted Risk (6-mo lead)",
         color="steelblue", linewidth=1.8)
ax0.axhline(0.4, color="red", linestyle="--", alpha=0.6, label="0.4 threshold")
ax0.fill_between(eval_index, 0, 1,
                 where=rec_periods.values == 1,
                 color="gray", alpha=0.25, label="Actual recession")
ax0.set_ylim(0, 1)
ax0.set_ylabel("Probability")
ax0.set_title("Recession Probability Forecast — Walk-Forward OOS (Calibrated)")
ax0.legend(loc="upper left")
ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Precision Recall curve
ax1 = axes[1]
precision, recall, _ = precision_recall_curve(y_eval, p_eval)
baseline = y_eval.mean()
ax1.plot(recall, precision, color="darkorange", linewidth=2,
         label=f"PR curve (AUC = {pr_auc:.3f})")
ax1.axhline(baseline, color="gray", linestyle="--", alpha=0.7,
            label=f"Baseline (prevalence = {baseline:.3f})")
ax1.set_xlabel("Recall")
ax1.set_ylabel("Precision")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_title("Precision-Recall Curve (OOS)")
ax1.legend(loc="upper right")

# Sahm Rule
ax2 = axes[2]
sahm = data_clean["sahm_rule"][mask]
ax2.plot(eval_index, sahm, color="seagreen", linewidth=1.8, label="Sahm Rule value")
ax2.axhline(0.5, color="black", linestyle=":", label="Sahm threshold (0.5)")
ax2.fill_between(eval_index, 0, sahm.clip(lower=0),
                 where=sahm > 0.5,
                 color="salmon", alpha=0.4, label="Above threshold")
ax2.set_title("Sahm Rule Indicator (Labour Market Stress)")
ax2.legend(loc="upper left")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
out_path = "recession_forecast.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nChart saved → {out_path}")