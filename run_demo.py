"""
Interactive Demonstration Script.
Scores two realistic borrowers (Prime vs Subprime) and showcases
the FICO score calculation and TreeSHAP Adverse Action explanations.
"""

import sys
import json
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from engine import CreditDecisionEngine


def main():
    print("=" * 70)
    print("FINTECH CREDIT RISK DECISIONING ENGINE - LIVE DEMO")
    print("=" * 70)
    
    engine = CreditDecisionEngine()
    print("[+] Model and TreeSHAP Explainer loaded successfully into memory.\n")
    
    # -------------------------------------------------------------
    # Case 1: Prime Applicant (High limit, punctual payments)
    # -------------------------------------------------------------
    prime_applicant = {
        "limit_balance": 150000.0,
        "sex": 1,
        "education": 1,
        "marriage": 2,
        "age": 35,
        "repayment_status_sept": -1,
        "repayment_status_aug": -1,
        "repayment_status_jul": -1,
        "repayment_status_jun": -1,
        "repayment_status_may": -1,
        "repayment_status_apr": -1,
        "bill_amt_sept": 4000.0,
        "bill_amt_aug": 3800.0,
        "bill_amt_jul": 3500.0,
        "bill_amt_jun": 3200.0,
        "bill_amt_may": 3000.0,
        "bill_amt_apr": 2800.0,
        "pay_amt_sept": 4000.0,
        "pay_amt_aug": 3800.0,
        "pay_amt_jul": 3500.0,
        "pay_amt_jun": 3200.0,
        "pay_amt_may": 3000.0,
        "pay_amt_apr": 2800.0
    }
    
    print("-" * 70)
    print("APPLICANT 1: PRIME BORROWER PROFILE")
    print("  Limit: $150,000 | Punctual Payments (Paid Duly) | Low Utilization (~2.6%)")
    print("-" * 70)
    result_1 = engine.score_applicant(prime_applicant)
    print(f"  Credit Score            : {result_1['credit_score']} / 850")
    print(f"  Probability of Default  : {result_1['probability_of_default'] * 100:.2f}%")
    print(f"  Risk Tier               : {result_1['risk_tier']}")
    print(f"  Underwriting Decision   : {result_1['decision']}")
    print(f"  Action                  : {result_1['recommended_action']}")
    print()
    
    # -------------------------------------------------------------
    # Case 2: Subprime Applicant (Low limit, multiple delinquencies)
    # -------------------------------------------------------------
    subprime_applicant = {
        "limit_balance": 15000.0,
        "sex": 2,
        "education": 3,
        "marriage": 1,
        "age": 24,
        "repayment_status_sept": 2,  # 2 months delay
        "repayment_status_aug": 2,
        "repayment_status_jul": 1,
        "repayment_status_jun": 1,
        "repayment_status_may": 0,
        "repayment_status_apr": 0,
        "bill_amt_sept": 16500.0,   # Over-limit
        "bill_amt_aug": 15800.0,
        "bill_amt_jul": 15200.0,
        "bill_amt_jun": 14000.0,
        "bill_amt_may": 12000.0,
        "bill_amt_apr": 10000.0,
        "pay_amt_sept": 0.0,
        "pay_amt_aug": 0.0,
        "pay_amt_jul": 500.0,
        "pay_amt_jun": 500.0,
        "pay_amt_may": 500.0,
        "pay_amt_apr": 500.0
    }
    
    print("-" * 70)
    print("APPLICANT 2: SUBPRIME / HIGH RISK BORROWER PROFILE")
    print("  Limit: $15,000 | 2 Months Past Due | Maxed-out Balance ($16,500) | $0 Paid in Sept")
    print("-" * 70)
    result_2 = engine.score_applicant(subprime_applicant)
    print(f"  Credit Score            : {result_2['credit_score']} / 850")
    print(f"  Probability of Default  : {result_2['probability_of_default'] * 100:.2f}%")
    print(f"  Risk Tier               : {result_2['risk_tier']}")
    print(f"  Underwriting Decision   : {result_2['decision']}")
    print(f"  Action                  : {result_2['recommended_action']}")
    print("\n  [REGULATORY ADVERSE ACTION NOTICES (TreeSHAP Explicability)]:")
    for i, reason in enumerate(result_2['adverse_action_reasons'], 1):
        print(f"    {i}. {reason['reason']} (SHAP Risk Impact: +{reason['impact']:.3f})")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
