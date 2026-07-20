from pathlib import Path

import pandas as pd

data_path = Path(__file__).resolve().parents[2] / "data" / "asfrVV_1950_to_2023.txt"
data = pd.read_csv(data_path, sep=r"\s+", header=None, names=["geo", "year", "age", "rate"])
data["geo"] = data["geo"].astype(int)
data["year"] = data["year"].astype(int)
data["age"] = data["age"].astype(int)

pd.set_option("display.width", 100)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

print(f"Shape: {data.shape[0]:,} rows x {data.shape[1]} columns\n")

print("First rows:")
print(data.head(10).to_string(index=False))

print("\nRanges:")
print(f"  geo : {data['geo'].min()} to {data['geo'].max()}  ({data['geo'].nunique()} unique)")
print(f"  year: {data['year'].min()} to {data['year'].max()}  ({data['year'].nunique()} unique)")
print(f"  age : {data['age'].min()} to {data['age'].max()}  ({data['age'].nunique()} unique)")
print(f"  rate: {data['rate'].min():.6f} to {data['rate'].max():.6f}")

print("\nRate summary:")
print(data["rate"].describe().to_string())

n_missing = data.isna().sum().sum()
n_zero_rate = (data["rate"] == 0).sum()
print(f"\nMissing values: {n_missing}")
print(f"Zero rates: {n_zero_rate:,} ({100 * n_zero_rate / len(data):.1f}%)")
