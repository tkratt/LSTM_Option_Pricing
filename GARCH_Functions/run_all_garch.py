import pandas as pd
import numpy as np
from arch import arch_model

# Load data
filename = input("Enter CSV filename (from Data/Assets/): ")
filepath = f"../Data/Assets/{filename}"

raw = pd.read_csv(filepath, nrows=5)
if raw.iloc[0].astype(str).str.contains("Ticker|^[A-Z]=F|\\^").any():
    skip = [1]
    if raw.iloc[1].astype(str).str.contains("^Date$").any():
        skip = [1, 2]
    df = pd.read_csv(filepath, skiprows=skip)
else:
    df = pd.read_csv(filepath)

df.rename(columns={df.columns[0]: "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

print(f"\nAvailable columns: {list(df.columns[1:])}")
price_col = input("Enter column name(s), comma-separated for DCC: ")
cols = [c.strip() for c in price_col.split(",")]
multi_asset = len(cols) > 1

# Process first/only asset
prices = pd.to_numeric(df[cols[0]], errors="coerce").dropna()
prices = prices[prices > 0]

# Compute log returns if needed
if prices.abs().mean() > 1:
    log_ret = np.log(prices / prices.shift(1)).dropna()
else:
    log_ret = prices.dropna()

scaled = log_ret * 100

# EGARCH(1,1) - Student-t
print("\n" + "="*60)
print("EGARCH(1,1) - Student-t")
print("="*60)
egarch = arch_model(scaled, vol="EGARCH", p=1, o=1, q=1, mean="ARX", dist="t")
egarch_res = egarch.fit(disp="off")
print(egarch_res.summary())

# GARCH(1,1) - GED
print("\n" + "="*60)
print("GARCH(1,1) - GED")
print("="*60)
ged = arch_model(scaled, vol="Garch", p=1, q=1, dist="ged")
ged_res = ged.fit(disp="off")
print(ged_res.summary())

# GJR-GARCH(1,1) - Student-t
print("\n" + "="*60)
print("GJR-GARCH(1,1) - Student-t")
print("="*60)
gjr = arch_model(scaled, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t")
gjr_res = gjr.fit(disp="off")
print(gjr_res.summary())

# DCC (multi-asset only)
if multi_asset:
    print("\n" + "="*60)
    print(f"DCC - Rolling 60-Day Correlation ({cols[0]} vs {cols[1]})")
    print("="*60)

    all_returns = pd.DataFrame()
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        s = s[s > 0]
        if s.abs().mean() > 1:
            all_returns[c] = np.log(s / s.shift(1))
        else:
            all_returns[c] = s
    all_returns = all_returns.dropna()

    rolling_corr = all_returns[cols[0]].rolling(window=60).corr(all_returns[cols[1]]).dropna()
    print(f"Mean correlation: {rolling_corr.mean():.4f}")
    print(f"Min correlation:  {rolling_corr.min():.4f}")
    print(f"Max correlation:  {rolling_corr.max():.4f}")
else:
    print("\n(DCC skipped — requires multiple assets)")

# Leaderboard
print("\n" + "="*60)
print("LEADERBOARD")
print("="*60)
results = {
    "EGARCH-t": egarch_res,
    "GARCH-GED": ged_res,
    "GJR-GARCH-t": gjr_res
}
board = pd.DataFrame({
    name: {"AIC": r.aic, "BIC": r.bic, "Log-Likelihood": r.loglikelihood}
    for name, r in results.items()
}).T.sort_values("AIC")
print(board.to_string())
print(f"\nBest model by AIC: {board.index[0]}")