"""
Model Training & Explainability Pipeline.
Trains Baseline (Logistic Regression) vs Champion (LightGBM) with financial metrics:
ROC-AUC, Gini Coefficient, KS-Statistic, Brier Score, and serializes TreeSHAP Explainer.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from features import compute_financial_features, FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "credit_card_defaults.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ARTIFACT_PATH = MODEL_DIR / "credit_model_bundle.joblib"


def calculate_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculates Kolmogorov-Smirnov (KS) statistic."""
    proba_defaulters = y_proba[y_true == 1]
    proba_non_defaulters = y_proba[y_true == 0]
    ks_stat, _ = ks_2samp(proba_defaulters, proba_non_defaulters)
    return float(ks_stat)


def train_pipeline():
    print(f"Loading raw dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    print("Applying financial feature engineering...")
    df_engineered = compute_financial_features(df)
    
    X = df_engineered[FEATURE_COLUMNS].copy()
    y = df_engineered["is_default"].values
    
    # Store feature medians for production imputation
    feature_medians = X.median().to_dict()
    X.fillna(feature_medians, inplace=True)
    
    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    
    # ==========================================
    # 1. Baseline Model: Logistic Regression
    # ==========================================
    print("\n--- Training Baseline (Logistic Regression) ---")
    baseline_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    baseline_pipe.fit(X_train, y_train)
    
    base_preds_proba = baseline_pipe.predict_proba(X_test)[:, 1]
    base_auc = roc_auc_score(y_test, base_preds_proba)
    base_gini = 2 * base_auc - 1
    base_ks = calculate_ks_statistic(y_test, base_preds_proba)
    base_brier = brier_score_loss(y_test, base_preds_proba)
    print(f"Baseline -> ROC-AUC: {base_auc:.4f} | Gini: {base_gini:.4f} | KS: {base_ks:.4f} | Brier: {base_brier:.4f}")

    # ==========================================
    # 2. Champion Model: LightGBM Classifier
    # ==========================================
    print("\n--- Training Champion (LightGBM Classifier) ---")
    champion_model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    champion_model.fit(X_train, y_train)
    
    champ_preds_proba = champion_model.predict_proba(X_test)[:, 1]
    champ_auc = roc_auc_score(y_test, champ_preds_proba)
    champ_gini = 2 * champ_auc - 1
    champ_ks = calculate_ks_statistic(y_test, champ_preds_proba)
    champ_prauc = average_precision_score(y_test, champ_preds_proba)
    champ_brier = brier_score_loss(y_test, champ_preds_proba)
    
    print(f"Champion -> ROC-AUC: {champ_auc:.4f} | Gini: {champ_gini:.4f} | KS: {champ_ks:.4f} | Brier: {champ_brier:.4f}")

    # ==========================================
    # 3. Explainability: TreeSHAP Explainer
    # ==========================================
    print("\nInitializing TreeSHAP Explainer...")
    # Use a representative sample of 100 background rows for fast baseline attribution
    background_sample = X_train.sample(n=min(100, len(X_train)), random_state=42)
    shap_explainer = shap.TreeExplainer(champion_model, data=background_sample)
    
    # Save Model Bundle
    bundle = {
        "model": champion_model,
        "baseline_model": baseline_pipe,
        "feature_columns": FEATURE_COLUMNS,
        "feature_medians": feature_medians,
        "metrics": {
            "baseline": {
                "roc_auc": round(base_auc, 4),
                "gini": round(base_gini, 4),
                "ks": round(base_ks, 4),
                "brier": round(base_brier, 4)
            },
            "champion": {
                "roc_auc": round(champ_auc, 4),
                "gini": round(champ_gini, 4),
                "ks": round(champ_ks, 4),
                "pr_auc": round(champ_prauc, 4),
                "brier": round(champ_brier, 4)
            }
        },
        "background_sample": background_sample,
        "explainer": shap_explainer
    }
    
    joblib.dump(bundle, MODEL_ARTIFACT_PATH)
    print(f"\nModel bundle saved successfully to {MODEL_ARTIFACT_PATH}")
    
    print("\n" + "=" * 55)
    print("FINANCIAL MODEL PERFORMANCE SUMMARY")
    print("=" * 55)
    print(f"{'Metric':<18} | {'Baseline (LogReg)':<18} | {'Champion (LightGBM)':<18}")
    print("-" * 55)
    print(f"{'ROC-AUC':<18} | {base_auc:<18.4f} | {champ_auc:<18.4f}")
    print(f"{'Gini Score':<18} | {base_gini:<18.4f} | {champ_gini:<18.4f}")
    print(f"{'KS Statistic':<18} | {base_ks:<18.4f} | {champ_ks:<18.4f}")
    print(f"{'Brier Score':<18} | {base_brier:<18.4f} | {champ_brier:<18.4f}")
    print("=" * 55)


if __name__ == "__main__":
    train_pipeline()
