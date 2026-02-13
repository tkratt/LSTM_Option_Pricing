import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Get user input
filename = input("Enter CSV filename (from Data/Assets/): ")
filepath = f"../Data/Assets/{filename}"

# Peek at file to detect format
raw = pd.read_csv(filepath, nrows=5)

# Skip ticker row if present (e.g. yfinance multi-header format)
if raw.iloc[0].astype(str).str.contains("Ticker|^[A-Z]=F|\\^").any():
    # Skip Ticker row, and Date row if present
    skip = [1]
    if raw.iloc[1].astype(str).str.contains("^Date$").any():
        skip = [1, 2]
    df = pd.read_csv(filepath, skiprows=skip)
else:
    df = pd.read_csv(filepath)

# Use first column as date regardless of name
df.rename(columns={df.columns[0]: "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Show available columns and let user pick
print(f"\nAvailable columns: {list(df.columns[1:])}")
price_col = input("Enter column name to run EGARCH on (price or returns): ")

# Compute log returns if needed
series = pd.to_numeric(df[price_col], errors="coerce").dropna()
if series.abs().mean() > 1:
    print(f"Detected prices — computing log returns from '{price_col}'")
    series = series[series > 0]
    log_ret = np.log(series / series.shift(1)).dropna()
else:
    print(f"Detected returns — using '{price_col}' directly")
    log_ret = series.dropna()

# Scale to percentage for arch library
returns = log_ret * 100

# Fit EGARCH(1,1)
model = arch_model(returns, vol="EGARCH", p=1, o=1, q=1, mean="ARX", dist="t")
result = model.fit(disp="off")
print(result.summary())

# Extract conditional volatility
dates = df.loc[log_ret.index, "Date"]
cond_vol = result.conditional_volatility / 100
ann_vol = cond_vol * np.sqrt(252)
