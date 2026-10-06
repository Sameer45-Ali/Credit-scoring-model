"""
Pydantic Request and Response Schemas for Credit Scoring API.
Provides strict data contracts, type enforcement, and realistic examples for OpenAPI docs.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class ApplicantProfile(BaseModel):
    limit_balance: float = Field(
        ...,
        gt=0,
        description="Amount of given credit limit (in USD or NT dollars)",
        json_schema_extra={"example": 50000.0}
    )
    sex: int = Field(
        ...,
        ge=1,
        le=2,
        description="Gender: 1 = Male, 2 = Female",
        json_schema_extra={"example": 1}
    )
    education: int = Field(
        ...,
        ge=1,
        le=4,
        description="Education: 1 = Grad School, 2 = University, 3 = High School, 4 = Others",
        json_schema_extra={"example": 2}
    )
    marriage: int = Field(
        ...,
        ge=1,
        le=3,
        description="Marital status: 1 = Married, 2 = Single, 3 = Others",
        json_schema_extra={"example": 2}
    )
    age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Applicant age in years (must be legal adult >= 18)",
        json_schema_extra={"example": 32}
    )
    repayment_status_sept: int = Field(
        default=0,
        ge=-2,
        le=9,
        description="Repayment status in Sept: -1 = Pay duly, 0 = Revolving, 1+ = Months delayed",
        json_schema_extra={"example": 0}
    )
    repayment_status_aug: int = Field(
        default=0, ge=-2, le=9, description="Repayment status in August", json_schema_extra={"example": 0}
    )
    repayment_status_jul: int = Field(
        default=0, ge=-2, le=9, description="Repayment status in July", json_schema_extra={"example": 0}
    )
    repayment_status_jun: int = Field(
        default=0, ge=-2, le=9, description="Repayment status in June", json_schema_extra={"example": 0}
    )
    repayment_status_may: int = Field(
        default=0, ge=-2, le=9, description="Repayment status in May", json_schema_extra={"example": 0}
    )
    repayment_status_apr: int = Field(
        default=0, ge=-2, le=9, description="Repayment status in April", json_schema_extra={"example": 0}
    )

    bill_amt_sept: float = Field(default=0.0, description="Bill statement amount in Sept", json_schema_extra={"example": 12000.0})
    bill_amt_aug: float = Field(default=0.0, description="Bill statement amount in August", json_schema_extra={"example": 11500.0})
    bill_amt_jul: float = Field(default=0.0, description="Bill statement amount in July", json_schema_extra={"example": 10200.0})
    bill_amt_jun: float = Field(default=0.0, description="Bill statement amount in June", json_schema_extra={"example": 9500.0})
    bill_amt_may: float = Field(default=0.0, description="Bill statement amount in May", json_schema_extra={"example": 8000.0})
    bill_amt_apr: float = Field(default=0.0, description="Bill statement amount in April", json_schema_extra={"example": 7200.0})

    pay_amt_sept: float = Field(default=0.0, ge=0.0, description="Payment amount in Sept", json_schema_extra={"example": 2500.0})
    pay_amt_aug: float = Field(default=0.0, ge=0.0, description="Payment amount in August", json_schema_extra={"example": 2000.0})
    pay_amt_jul: float = Field(default=0.0, ge=0.0, description="Payment amount in July", json_schema_extra={"example": 1800.0})
    pay_amt_jun: float = Field(default=0.0, ge=0.0, description="Payment amount in June", json_schema_extra={"example": 1500.0})
    pay_amt_may: float = Field(default=0.0, ge=0.0, description="Payment amount in May", json_schema_extra={"example": 1500.0})
    pay_amt_apr: float = Field(default=0.0, ge=0.0, description="Payment amount in April", json_schema_extra={"example": 1500.0})


class AdverseActionReason(BaseModel):
    feature: str
    impact: float
    reason: str


class CalculatedFeatures(BaseModel):
    current_utilization: float
    avg_utilization_6m: float
    current_pay_ratio: float
    num_delinquent_months_6m: int
    max_delinquency_6m: int


class ScoringResponse(BaseModel):
    credit_score: int = Field(..., description="FICO-equivalent score (300 - 850)", json_schema_extra={"example": 724})
    probability_of_default: float = Field(..., description="Model predicted default probability", json_schema_extra={"example": 0.062})
    decision: str = Field(..., description="Automated decision policy", json_schema_extra={"example": "APPROVED_WITH_CONDITIONS"})
    risk_tier: str = Field(..., description="Credit risk tier", json_schema_extra={"example": "Tier 2 (Near-Prime)"})
    recommended_action: str = Field(..., description="Underwriting guidance")
    adverse_action_reasons: List[AdverseActionReason] = Field(
        default=[], description="Top SHAP-derived adverse risk factors"
    )
    calculated_features: CalculatedFeatures


class BatchApplicantsRequest(BaseModel):
    applicants: List[ApplicantProfile]


class BatchScoringResponse(BaseModel):
    total_processed: int
    results: List[ScoringResponse]


class ModelInfoResponse(BaseModel):
    model_name: str
    champion_algorithm: str
    baseline_algorithm: str
    metrics: Dict[str, Any]
    feature_count: int
    status: str
