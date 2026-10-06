"""
Unit tests for FinTech Decision Engine and Scoring Logic.
"""

import sys
from pathlib import Path
import pytest

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from engine import CreditDecisionEngine


@pytest.fixture(scope="module")
def engine():
    return CreditDecisionEngine()


def test_score_mapping(engine):
    # Very low PD (0.01) should map to a high prime score (>750)
    prime_score = engine.probability_to_credit_score(0.01)
    assert 750 <= prime_score <= 850

    # Very high PD (0.60) should map to a low subprime score (<550)
    subprime_score = engine.probability_to_credit_score(0.60)
    assert 300 <= subprime_score <= 580


def test_prime_applicant_scoring(engine):
    # Prime customer: large limit, zero delinquency, pays full statement
    prime_data = {
        "limit_balance": 200000.0,
        "sex": 1,
        "education": 1,
        "marriage": 2,
        "age": 38,
        "repayment_status_sept": 0,
        "repayment_status_aug": 0,
        "repayment_status_jul": 0,
        "repayment_status_jun": 0,
        "repayment_status_may": 0,
        "repayment_status_apr": 0,
        "bill_amt_sept": 5000.0,
        "bill_amt_aug": 4800.0,
        "bill_amt_jul": 4200.0,
        "bill_amt_jun": 4000.0,
        "bill_amt_may": 3900.0,
        "bill_amt_apr": 3500.0,
        "pay_amt_sept": 5000.0,
        "pay_amt_aug": 4800.0,
        "pay_amt_jul": 4200.0,
        "pay_amt_jun": 4000.0,
        "pay_amt_may": 3900.0,
        "pay_amt_apr": 3500.0
    }
    result = engine.score_applicant(prime_data)
    assert result["credit_score"] >= 670
    assert result["decision"] in ["APPROVED", "APPROVED_WITH_CONDITIONS"]
    assert result["probability_of_default"] < 0.25


def test_high_risk_applicant_scoring_and_adverse_reasons(engine):
    # Subprime customer: small limit, heavily delinquent (2-3 months behind), near zero payments
    risky_data = {
        "limit_balance": 10000.0,
        "sex": 2,
        "education": 3,
        "marriage": 1,
        "age": 25,
        "repayment_status_sept": 3,
        "repayment_status_aug": 2,
        "repayment_status_jul": 2,
        "repayment_status_jun": 1,
        "repayment_status_may": 1,
        "repayment_status_apr": 0,
        "bill_amt_sept": 12000.0,
        "bill_amt_aug": 11500.0,
        "bill_amt_jul": 11000.0,
        "bill_amt_jun": 10500.0,
        "bill_amt_may": 10000.0,
        "bill_amt_apr": 9800.0,
        "pay_amt_sept": 0.0,
        "pay_amt_aug": 0.0,
        "pay_amt_jul": 0.0,
        "pay_amt_jun": 0.0,
        "pay_amt_may": 0.0,
        "pay_amt_apr": 0.0
    }
    result = engine.score_applicant(risky_data)
    assert result["credit_score"] < 640
    assert result["decision"] in ["REJECTED", "MANUAL_REVIEW"]
    # Verify SHAP adverse reasons were produced
    assert len(result["adverse_action_reasons"]) > 0
    top_reason = result["adverse_action_reasons"][0]
    assert "reason" in top_reason
    assert "impact" in top_reason
