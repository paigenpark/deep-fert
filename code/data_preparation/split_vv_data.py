"""Preprocess HFD vertical-parallelogram (VV) ASFR data into the DL model format.

The VV file groups events by calendar year and age reached during the year (ARDY)
-- the F2 Lexis shape -- as opposed to the RR file's Lexis squares (completed age,
ACY). This mirrors split_period_data.py but for the VV file's 5-column layout:

    Code   Year   ARDY   Cohort   ASFR         (raw asfr/asfrVV.txt)
      |      |      |       |        |
   country  year  age(ARDY) (drop) rate   ->   [country_idx, year, age, rate]

The Cohort column is redundant (Cohort == Year - ARDY) and is dropped. Country
codes are mapped through the SHARED geos_key.npy written by split_period_data.py,
so index 0 means the same country (AUT) in both the square and parallelogram
datasets -- keeping the two data types directly comparable.

Output: data/asfrVV_1950_to_2023.txt  (same numeric format as asfr_1950_to_2023.txt)
"""

import os

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

RAW_VV = "../../data/asfr/asfrVV.txt"
GEOS_KEY = "../../data/geos_key.npy"
OUT = "../../data/asfrVV_1950_to_2023.txt"

# Reuse the country->index mapping from the square-data preprocessing so the two
# datasets share a consistent geography index.
geos_key = np.load(GEOS_KEY)
code_to_index = {code: int(idx) for code, idx in geos_key}

data = []
unmapped = set()

with open(RAW_VV, "r") as file:
    for row_index, row in enumerate(file):
        row = row.strip()
        if row_index <= 2:  # 3 header lines
            print(row)
            continue

        columns = row.split()
        # VV layout: Code Year ARDY Cohort ASFR
        country, year, ardy, _cohort, rate = columns
        year = int(year)

        # ARDY plays the role of "age"; non-numeric labels ("12-", "55+") -> drop
        try:
            age = int(ardy)
        except ValueError:
            age = -1

        try:
            rate = float(rate)
        except ValueError:
            rate = -1
        if rate > 1:
            rate = 1

        if country not in code_to_index:
            unmapped.add(country)
            continue
        country_idx = code_to_index[country]

        if age != -1 and rate != -1 and age <= 54:
            data.append([country_idx, year, age, rate])

asfr_vv_data = np.array(data)
print("ASFR VV data shape:", asfr_vv_data.shape)
print(f"Years: {int(asfr_vv_data[:, 1].min())}-{int(asfr_vv_data[:, 1].max())}")
print(f"Ages (ARDY): {int(asfr_vv_data[:, 2].min())}-{int(asfr_vv_data[:, 2].max())}")
print(f"Countries: {len(np.unique(asfr_vv_data[:, 0]))}")
if unmapped:
    print(f"WARNING: {len(unmapped)} country codes not in geos_key, skipped: {sorted(unmapped)}")

np.savetxt(OUT, asfr_vv_data)
print(f"Saved: {OUT}")
