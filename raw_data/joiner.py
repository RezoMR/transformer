import pandas as pd

# Zoznam CSV súborov (môžu byť aj cesty)
csv_files = [
    "aFRR.csv",
    "mFRR.csv",
    "FCR.csv",
    "odchylka.csv",
    "complex_imbalance_2025_regulardaily.csv"
]
dfs = [pd.read_csv(f) for f in csv_files]

# Spojí DataFrame do šírky (vedľa seba)
merged = pd.concat(dfs, axis=1)
start = pd.Timestamp("2025-01-01 00:15")
merged["period_end"] = start + pd.to_timedelta(merged.index * 15, unit="min")

merged.to_csv("raw_data_merged.csv", index=False, encoding="utf-8")
