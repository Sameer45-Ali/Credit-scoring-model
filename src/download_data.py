"""
Download and prepare the UCI Credit Card Default dataset (30,000 real client records).
Maps raw variable codes to standard, intuitive financial domain names.
"""

from pathlib import Path
from sklearn.datasets import fetch_openml
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_FILE = DATA_DIR / "credit_card_defaults.csv"

# Human-readable mapping of UCI features
COLUMN_MAPPING = {
    "x1": "limit_balance",
    "x2": "sex",
    "x3": "education",
    "x4": "marriage",
    "x5": "age",
    "x6": "repayment_status_sept",
    "x7": "repayment_status_aug",
    "x8": "repayment_status_jul",
    "x9": "repayment_status_jun",
    "x10": "repayment_status_may",
    "x11": "repayment_status_apr",
    "x12": "bill_amt_sept",
    "x13": "bill_amt_aug",
    "x14": "bill_amt_jul",
    "x15": "bill_amt_jun",
    "x16": "bill_amt_may",
    "x17": "bill_amt_apr",
    "x18": "pay_amt_sept",
    "x19": "pay_amt_aug",
    "x20": "pay_amt_jul",
    "x21": "pay_amt_jun",
    "x22": "pay_amt_may",
    "x23": "pay_amt_apr",
    "y": "is_default"
}

def download_and_save():
    print("Fetching 'default-of-credit-card-clients' from OpenML...")
    dataset = fetch_openml("default-of-credit-card-clients", version=1, as_frame=True)
    df = dataset.frame.copy()
    
    # Rename columns
    df.rename(columns=COLUMN_MAPPING, inplace=True)
    
    # Ensure proper data types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df["is_default"] = df["is_default"].astype(int)
    
    df.to_csv(RAW_FILE, index=False)
    print(f"Dataset successfully saved to {RAW_FILE}")
    print(f"Shape: {df.shape}")
    print(f"Default rate: {df['is_default'].mean():.2%}")
    print(df.head(2))

if __name__ == "__main__":
    download_and_save()
