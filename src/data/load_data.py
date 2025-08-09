from sklearn.datasets import fetch_california_housing
import pandas as pd
from pathlib import Path

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    out_path = OUT / "california_housing_raw.csv"
    df.to_csv(out_path, index=False)
    print("Wrote:", out_path)