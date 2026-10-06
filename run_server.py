"""
Simple script to launch the Credit Scoring API server.
"""

import sys
from pathlib import Path
import uvicorn

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  LAUNCHING FINTECH CREDIT RISK SCORING API")
    print("=" * 60)
    print("  --> Local API:   http://127.0.0.1:8000")
    print("  --> Swagger UI:  http://127.0.0.1:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)
