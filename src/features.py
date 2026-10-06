"""
FinTech Feature Engineering Module.
Transforms raw transaction statements, bill amounts, and delinquency history
into behavioral risk ratios (Utilization, Payment Ratios, Delinquency counts).
"""

import numpy as np
import pandas as pd


def compute_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes advanced banking risk features from raw credit history."""
    data = df.copy()

    # 1. Credit Utilization Ratios (Current & 6-month aggregate)
    # Clip utilization to realistic range [0, 5] to avoid extreme outliers
    bill_cols = [
        "bill_amt_sept", "bill_amt_aug", "bill_amt_jul",
        "bill_amt_jun", "bill_amt_may", "bill_amt_apr"
    ]
    pay_cols = [
        "pay_amt_sept", "pay_amt_aug", "pay_amt_jul",
        "pay_amt_jun", "pay_amt_may", "pay_amt_apr"
    ]
    status_cols = [
        "repayment_status_sept", "repayment_status_aug", "repayment_status_jul",
        "repayment_status_jun", "repayment_status_may", "repayment_status_apr"
    ]

    limit = np.maximum(data["limit_balance"], 1.0)
    data["current_utilization"] = np.clip(data["bill_amt_sept"] / limit, 0.0, 5.0)

    # Average bill and max bill over 6 months
    avg_bill = data[bill_cols].mean(axis=1)
    max_bill = data[bill_cols].max(axis=1)
    data["avg_utilization_6m"] = np.clip(avg_bill / limit, 0.0, 5.0)
    data["max_utilization_6m"] = np.clip(max_bill / limit, 0.0, 5.0)

    # 2. Payment-to-Bill Ratios (Liquidity & Repayment Ability)
    denom = np.maximum(data["bill_amt_sept"], 1.0)
    data["current_pay_ratio"] = np.clip(data["pay_amt_sept"] / denom, 0.0, 5.0)

    total_bills = np.maximum(data[bill_cols].sum(axis=1), 1.0)
    total_pays = data[pay_cols].sum(axis=1)
    data["avg_pay_ratio_6m"] = np.clip(total_pays / total_bills, 0.0, 5.0)

    # 3. Delinquency Behavior Metrics
    # In UCI dataset: <=0 means paid on time / no delay; >0 indicates months of delay
    data["max_delinquency_6m"] = data[status_cols].max(axis=1)
    data["num_delinquent_months_6m"] = (data[status_cols] > 0).sum(axis=1)
    data["recent_delinquency_flag"] = (
        (data["repayment_status_sept"] > 0) | (data["repayment_status_aug"] > 0)
    ).astype(int)

    # 4. Trajectory Trends (Are debts expanding? Are payments collapsing?)
    data["debt_trend_6m"] = data["bill_amt_sept"] - data["bill_amt_apr"]
    data["payment_trend_6m"] = data["pay_amt_sept"] - data["pay_amt_apr"]

    return data


# Standard training feature list (domain-meaningful features only)
FEATURE_COLUMNS = [
    "limit_balance",
    "age",
    "education",
    "marriage",
    "current_utilization",
    "avg_utilization_6m",
    "max_utilization_6m",
    "current_pay_ratio",
    "avg_pay_ratio_6m",
    "max_delinquency_6m",
    "num_delinquent_months_6m",
    "recent_delinquency_flag",
    "debt_trend_6m",
    "payment_trend_6m",
    "repayment_status_sept",
    "repayment_status_aug",
    "bill_amt_sept",
    "pay_amt_sept"
]
