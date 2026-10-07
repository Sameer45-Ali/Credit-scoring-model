"""
Integration tests for FastAPI endpoints using TestClient.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Credit Risk Underwriting" in response.text


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["engine_ready"] is True


def test_model_info(client):
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["champion_algorithm"].startswith("LightGBM")
    assert "metrics" in data
    assert "champion" in data["metrics"]
    assert data["metrics"]["champion"]["roc_auc"] > 0.70


def test_score_valid_applicant(client):
    payload = {
        "limit_balance": 80000.0,
        "sex": 1,
        "education": 2,
        "marriage": 2,
        "age": 30,
        "repayment_status_sept": 0,
        "repayment_status_aug": 0,
        "repayment_status_jul": 0,
        "repayment_status_jun": 0,
        "repayment_status_may": 0,
        "repayment_status_apr": 0,
        "bill_amt_sept": 5000.0,
        "bill_amt_aug": 4500.0,
        "bill_amt_jul": 4000.0,
        "bill_amt_jun": 3500.0,
        "bill_amt_may": 3000.0,
        "bill_amt_apr": 2500.0,
        "pay_amt_sept": 2000.0,
        "pay_amt_aug": 1500.0,
        "pay_amt_jul": 1500.0,
        "pay_amt_jun": 1000.0,
        "pay_amt_may": 1000.0,
        "pay_amt_apr": 1000.0
    }
    response = client.post("/api/v1/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 300 <= data["credit_score"] <= 850
    assert 0.0 <= data["probability_of_default"] <= 1.0
    assert "decision" in data
    assert "risk_tier" in data
    assert "calculated_features" in data


def test_validation_error_on_underage_applicant(client):
    # Applicant younger than 18 must be blocked by Pydantic schema
    invalid_payload = {
        "limit_balance": 50000.0,
        "sex": 1,
        "education": 2,
        "marriage": 2,
        "age": 16  # Invalid (<18)
    }
    response = client.post("/api/v1/score", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity
