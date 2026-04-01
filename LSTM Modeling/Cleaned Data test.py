import pandas as pd
import numpy as np

# Load your cleaned data
print("Loading data...")
df = pd.read_csv("Cleaned_SPX_Options.csv")

print("\n--- SPX Options Data Quality Report ---")
print(f"Total rows to test: {len(df):,}")

# 1. Missing or Infinite Values
# LSTMs will instantly crash if they encounter NaNs or Infs
missing_values = df.isnull().sum().sum()
infinite_values = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
print(f"\n[1] Missing Values (NaNs): {missing_values}")
print(f"[2] Infinite Values (Infs): {infinite_values}")

# 2. Price and Volatility Logic
# Prices and Implied Volatility cannot be negative
negative_prices = len(df[(df['C_MID'] < 0) | (df['P_MID'] < 0)])
zero_or_negative_iv = len(df[(df['C_IV'] <= 0) | (df['P_IV'] <= 0)])
print(f"[3] Negative Mid Prices: {negative_prices}")
print(f"[4] Zero or Negative Implied Volatility: {zero_or_negative_iv}")

# 3. Hard Arbitrage Boundaries
# A call option's price can NEVER exceed the underlying asset's price ($C \le S$)
call_exceeds_spot = len(df[df['C_MID'] > df['UNDERLYING_LAST']])

# A put option's price can NEVER exceed the strike price ($P \le K$)
put_exceeds_strike = len(df[df['P_MID'] > df['STRIKE']])

print(f"[5] Calls priced higher than SPX: {call_exceeds_spot}")
print(f"[6] Puts priced higher than Strike: {put_exceeds_strike}")

# 4. Intrinsic Value Bounds (Soft Check)
# An option should generally not trade below its intrinsic value.
# For calls: $C \ge \max(0, S - K)$
# For puts: $P \ge \max(0, K - S)$
# Note: Deep ITM European options can slightly violate this due to high interest rates,
# but massive violations indicate bad data.
call_below_intrinsic = len(df[df['C_MID'] < (df['UNDERLYING_LAST'] - df['STRIKE']) - 0.50]) # $0.50 buffer
put_below_intrinsic = len(df[df['P_MID'] < (df['STRIKE'] - df['UNDERLYING_LAST']) - 0.50])

print(f"[7] Calls drastically below intrinsic value: {call_below_intrinsic}")
print(f"[8] Puts drastically below intrinsic value: {put_below_intrinsic}")

# --- CONCLUSION ---
total_errors = (missing_values + infinite_values + negative_prices +
                zero_or_negative_iv + call_exceeds_spot + put_exceeds_strike)

print("\n--- Final Verdict ---")
if total_errors == 0:
    print("PASS: The dataset is mathematically sound and ready for the LSTM.")
else:
    print(f"FAIL: Found {total_errors} hard errors. You must filter these out before training.")

    import pandas as pd

    print("Loading data for Phase 2 cleaning...")
    df = pd.read_csv("Cleaned_SPX_Options.csv")
    original_len = len(df)

    # 1. Drop the NaNs instantly
    df = df.dropna()

    # 2. Drop any busted Implied Volatilities
    df = df[(df['C_IV'] > 0) & (df['P_IV'] > 0)]

    # 3. Filter for Moneyness (Spot / Strike)
    # We only want to train the model on options within +/- 20% of the current SPX price.
    # This eliminates the deep ITM/OTM garbage causing the intrinsic value violations.
    df['MONEYNESS'] = df['UNDERLYING_LAST'] / df['STRIKE']
    df = df[(df['MONEYNESS'] >= 0.8) & (df['MONEYNESS'] <= 1.2)]

    # 4. Hard boundary enforcement
    # Just in case any weird spreads snuck through the moneyness filter
    df = df[df['C_MID'] >= (df['UNDERLYING_LAST'] - df['STRIKE'])]
    df = df[df['P_MID'] >= (df['STRIKE'] - df['UNDERLYING_LAST'])]

    # Save the truly clean data
    df.to_csv("LSTM_Ready_SPX_Options.csv", index=False)

    print(f"Original Rows: {original_len:,}")
    print(f"Final Clean Rows: {len(df):,}")
    print(f"Removed {original_len - len(df):,} noisy rows.")