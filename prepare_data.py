"""
prepare_data.py
Merges indy500_results.csv and weather_data.csv, adds a technology_index
column, drops incomplete rows, and saves the result as combined_data.csv.
"""

import pandas as pd

# ── Load ───────────────────────────────────────────────────────────────────────
results = pd.read_csv("data/indy500_results.csv")
weather = pd.read_csv("data/weather_data.csv")

# ── Merge on year ──────────────────────────────────────────────────────────────
df = results.merge(weather, on="year", how="left")

# ── Technology index ───────────────────────────────────────────────────────────
# Linear scale from 0.0 (1911) to 1.0 (2024), computed from the full
# historical range so the value is stable regardless of which rows survive
# the cleaning step below.
YEAR_MIN = 1911
YEAR_MAX = 2024
df["technology_index"] = (df["year"] - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)
df["technology_index"] = df["technology_index"].round(4)

# ── Clean ──────────────────────────────────────────────────────────────────────
rows_before = len(df)
df = df.dropna()
rows_dropped = rows_before - len(df)

# ── Save ───────────────────────────────────────────────────────────────────────
df.to_csv("data/combined_data.csv", index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"Rows before cleaning : {rows_before}")
print(f"Rows dropped (NaN)   : {rows_dropped}  "
      f"(races before 1940 lack ERA5 weather data)")
print(f"Rows in combined_data: {len(df)}")
print(f"\nYear range  : {int(df['year'].min())} – {int(df['year'].max())}")
print(f"\nColumns ({len(df.columns)}):")
for col in df.columns:
    print(f"  {col}")
print("\nSaved to data/combined_data.csv")
