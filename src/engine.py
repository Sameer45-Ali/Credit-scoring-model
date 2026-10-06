"""
Credit Risk Decision Engine.
Implements:
1. Probability of Default (PD) scoring.
2. FICO-style Credit Score transformation (300 - 850) using Points to Double Odds (PDO).
3. Risk Tier classification & Automated Decisioning policy.
4. TreeSHAP Adverse Action explanations for FCRA / regulatory compliance.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import joblib
import numpy as np
import pandas as pd

from features import compute_financial_features, FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_ARTIFACT_PATH = BASE_DIR / "models" / "credit_model_bundle.joblib"

# Human-readable mapping for regulatory adverse action notices
ADVERSE_REASON_MAP = {
    "current_utilization": "High current credit card utilization ratio",
    "avg_utilization_6m": "High average credit line utilization over the past 6 months",
    "max_utilization_6m": "Excessive peak credit limit utilization",
    "max_delinquency_6m": "Severe payment delinquency recorded in prior months",
    "num_delinquent_months_6m": "Frequent past-due payments across multiple billing cycles",
    "recent_delinquency_flag": "Recent delinquent payment status in the last 60 days",
    "current_pay_ratio": "Low payment amount relative to current bill statement",
    "avg_pay_ratio_6m": "Low historical repayment-to-debt ratio",
    "limit_balance": "Low revolving credit limit assigned",
    "debt_trend_6m": "Accelerating debt balance growth over the last 6 months",
    "payment_trend_6m": "Declining payment amounts over the last 6 months",
    "repayment_status_sept": "Payment delay in the most recent billing cycle",
    "repayment_status_aug": "Payment delay in the previous billing cycle",
    "age": "Limited credit track record associated with age profile",
    "bill_amt_sept": "High outstanding statement balance"
}


class CreditDecisionEngine:
    def __init__(self, artifact_path: Optional[Path] = None):
        path = artifact_path or MODEL_ARTIFACT_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {path}. Run train.py first."
            )
        
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.baseline_model = bundle["baseline_model"]
        self.feature_columns = bundle["feature_columns"]
        self.feature_medians = bundle["feature_medians"]
        self.metrics = bundle["metrics"]
        self.explainer = bundle["explainer"]

    @staticmethod
    def probability_to_credit_score(
        pd_value: float,
        base_score: int = 650,
        base_odds: float = 4.0,
        pdo: int = 40
    ) -> int:
        """
        Maps Probability of Default (PD) to standard 300-850 credit score using PDO scaling.
        Formula: Score = BaseScore + (PDO / ln(2)) * (ln((1 - PD)/PD) - ln(base_odds))
        """
        eps = 1e-6
        pd_safe = np.clip(pd_value, eps, 1.0 - eps)
        odds = (1.0 - pd_safe) / pd_safe
        factor = pdo / np.log(2)
        score = base_score + factor * (np.log(odds) - np.log(base_odds))
        return int(np.clip(round(score), 300, 850))

    @staticmethod
    def evaluate_risk_tier(score: int, pd_value: float) -> Dict[str, str]:
        """Maps credit score to FinTech decision policy & risk grade."""
        if score >= 740:
            return {
                "decision": "APPROVED",
                "tier": "Tier 1 (Prime)",
                "recommended_action": "Instant approval at prime interest rates."
            }
        elif score >= 670:
            return {
                "decision": "APPROVED_WITH_CONDITIONS",
                "tier": "Tier 2 (Near-Prime)",
                "recommended_action": "Approve with standard interest rate or collateral limit."
            }
        elif score >= 580:
            return {
                "decision": "MANUAL_REVIEW",
                "tier": "Tier 3 (Fair / Elevated Risk)",
                "recommended_action": "Refer to credit underwriting analyst for income/employment verification."
            }
        else:
            return {
                "decision": "REJECTED",
                "tier": "Tier 4 (Subprime)",
                "recommended_action": "Application declined due to high projected probability of default."
            }

    def generate_adverse_action_reasons(
        self, features_df: pd.DataFrame, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Uses TreeSHAP values to find the exact features that increased default risk
        for this specific applicant (Regulatory requirement for Adverse Action notices).
        """
        shap_values = self.explainer.shap_values(features_df)
        
        # Binary classification LightGBM: take values for default class (class 1)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        # Positive SHAP value means pushing towards default (risk factor)
        feature_impacts = []
        for feature_name, impact in zip(self.feature_columns, sv):
            if impact > 0:  # Risk-increasing factor
                feature_impacts.append({
                    "feature": feature_name,
                    "impact": float(impact),
                    "reason": ADVERSE_REASON_MAP.get(
                        feature_name, f"Adverse risk factor: {feature_name}"
                    )
                })

        # Sort by impact descending
        feature_impacts.sort(key=lambda x: x["impact"], reverse=True)
        return feature_impacts[:top_k]

    def score_applicant(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """End-to-end evaluation of an individual credit applicant."""
        raw_df = pd.DataFrame([applicant_data])
        
        # Apply financial feature transformations
        engineered_df = compute_financial_features(raw_df)
        
        # Extract features and fill missing values with learned medians
        X = engineered_df[self.feature_columns].copy()
        X.fillna(self.feature_medians, inplace=True)
        
        # 1. Model inference
        proba = float(self.model.predict_proba(X)[0, 1])
        
        # 2. Score conversion
        credit_score = self.probability_to_credit_score(proba)
        
        # 3. Policy & risk tiering
        tier_info = self.evaluate_risk_tier(credit_score, proba)
        
        # 4. Explainability (SHAP Adverse Factors)
        adverse_reasons = self.generate_adverse_action_reasons(X, top_k=3)
        
        return {
            "credit_score": credit_score,
            "probability_of_default": round(proba, 4),
            "decision": tier_info["decision"],
            "risk_tier": tier_info["tier"],
            "recommended_action": tier_info["recommended_action"],
            "adverse_action_reasons": adverse_reasons,
            "calculated_features": {
                "current_utilization": round(float(X["current_utilization"].iloc[0]), 4),
                "avg_utilization_6m": round(float(X["avg_utilization_6m"].iloc[0]), 4),
                "current_pay_ratio": round(float(X["current_pay_ratio"].iloc[0]), 4),
                "num_delinquent_months_6m": int(X["num_delinquent_months_6m"].iloc[0]),
                "max_delinquency_6m": int(X["max_delinquency_6m"].iloc[0])
            }
        }
