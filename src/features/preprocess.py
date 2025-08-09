import pandas as pd
from pathlib import Path

IN = Path("data/raw/california_housing_raw.csv")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    df = pd.read_csv(IN)
    # example feature: rooms_per_household
    df["rooms_per_household"] = df["AveRooms"] / df["HouseAge"].replace(0, 1)
    df.to_csv(OUT / "california_housing_processed.csv", index=False)
    print("Processed ->", OUT / "california_housing_processed.csv")
