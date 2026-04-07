import math
from scipy.stats import norm


def merton_jump_call(S, K, T, r, sigma, lam, mu_J, sigma_J, N=50):
    """
    Prices a European call option using the Merton Jump Diffusion model.
    """
    sigma = np.maximum(sigma, 1e-8)
    T = np.maximum(T, 1e-8)

    # Expected percentage change from a jump
    k = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1

    # Risk-neutral jump intensity
    lam_prime = lam * (1 + k)

    price = 0.0

    # Summing the Black-Scholes prices weighted by Poisson probabilities
    for n in range(N):
        # Poisson probability of exactly 'n' jumps occurring
        weight = np.exp(-lam_prime * T) * ((lam_prime * T) ** n) / math.factorial(n)

        # Adjusted volatility and risk-free rate for 'n' jumps
        sigma_n = np.sqrt(sigma ** 2 + (n * sigma_J ** 2) / T)
        r_n = r - lam * k + (n * np.log(1 + k)) / T

        # Standard Black-Scholes d1 and d2 inside the sum
        d1 = (np.log(S / K) + (r_n + 0.5 * sigma_n ** 2) * T) / (sigma_n * np.sqrt(T))
        d2 = d1 - sigma_n * np.sqrt(T)

        bs_call_n = S * norm.cdf(d1) - K * np.exp(-r_n * T) * norm.cdf(d2)

        price += weight * bs_call_n

    return price
# multiple contracts EGARCH
import yfinance as yf
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
from arch import arch_model
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# ---------------------------------------------------------
# 1. Merton Jump Diffusion Pricing Function
# ---------------------------------------------------------
def merton_jump_call(S, K, T, r, sigma, lam, mu_J, sigma_J, N=50):
    sigma = np.maximum(sigma, 1e-8)
    T = np.maximum(T, 1e-8)
    k = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1
    lam_prime = lam * (1 + k)
    price = 0.0

    for n in range(N):
        weight = np.exp(-lam_prime * T) * ((lam_prime * T) ** n) / math.factorial(n)
        sigma_n = np.sqrt(sigma ** 2 + (n * sigma_J ** 2) / T)
        r_n = r - lam * k + (n * np.log(1 + k)) / T

        d1 = (np.log(S / K) + (r_n + 0.5 * sigma_n ** 2) * T) / (sigma_n * np.sqrt(T))
        d2 = d1 - sigma_n * np.sqrt(T)

        bs_call_n = S * norm.cdf(d1) - K * np.exp(-r_n * T) * norm.cdf(d2)
        price += weight * bs_call_n

    return price


# ---------------------------------------------------------
# 2. Load Options Data & Find Top Contracts
# ---------------------------------------------------------
print("Loading Real TSLA Options Data...")
df_options = pd.read_csv('/Users/teddy/LSTM Option Pricing/LSTM_ready_TSLA_Options.csv')

call_cols = [col for col in df_options.columns if not col.startswith('P_')]
df_calls = df_options[call_cols].copy()

df_calls['STRIKE'] = (df_calls['UNDERLYING_LAST'] / df_calls['MONEYNESS']).round(2)
df_calls['QUOTE_DATE'] = pd.to_datetime(df_calls['QUOTE_DATE']).dt.tz_localize(None)

# Dynamically find the top 20 contracts by row count (based on your screenshot)
contract_counts = df_calls.groupby(['EXPIRE_DATE', 'STRIKE']).size().reset_index(name='Row_Count')
top_contracts_df = contract_counts.sort_values(by='Row_Count', ascending=False).head(20)

# Create a list of tuples: [(expire1, strike1), (expire2, strike2), ...]
top_contracts = list(zip(top_contracts_df['EXPIRE_DATE'], top_contracts_df['STRIKE']))
print(f"Testing the top {len(top_contracts)} contracts...")

# ---------------------------------------------------------
# 3. Procure Underlying Data & Feature Engineering
# ---------------------------------------------------------
ticker = 'TSLA'
print(f"Fetching {ticker} data to train EGARCH Model...")
# Started earlier to ensure all 2022 options have enough historical runway
raw_data = yf.download(ticker, start='2015-01-01', end='2024-12-31')

df = pd.DataFrame()
df['Close'] = raw_data['Close']
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna()
df['Scaled_Return'] = df['Log_Return'] * 100.0

# Explicitly start the test set right before your earliest option (April 2022)
# This guarantees our EGARCH forecast dates will overlap with your options dates
test_start_date = '2022-01-01'
split_idx = df.index.get_loc(df[df.index >= test_start_date].index[0])
test_dates = df.index[split_idx:]

# ---------------------------------------------------------
# 4. EGARCH Model Architecture & Rolling Forecast
# ---------------------------------------------------------
print(f"Starting pure EGARCH rolling forecast for {len(df) - split_idx} test days...")

predicted_ann_vols = []

for i in range(len(df) - split_idx):
    train_window = df['Scaled_Return'].iloc[:split_idx + i]
    am = arch_model(train_window, vol='EGARCH', p=1, o=1, q=1, dist='Normal')
    res = am.fit(disp='off')

    forecasts = res.forecast(horizon=1, align='origin')
    pred_var_scaled = forecasts.variance.iloc[-1, 0]
    pred_vol_daily = np.sqrt(pred_var_scaled) / 100.0
    predicted_ann_vols.append(pred_vol_daily * np.sqrt(252))

    if (i + 1) % 100 == 0:
        print(f"Completed {i + 1} / {len(df) - split_idx} forecasts...")

vol_forecast_df = pd.DataFrame({
    'QUOTE_DATE': test_dates.tz_localize(None),
    'EGARCH_VOL': predicted_ann_vols
})

# ---------------------------------------------------------
# 5. Multi-Contract Validation & Pricing
# ---------------------------------------------------------
print("\nValidating Model against Real TSLA Option Prices...")

all_actual_prices = []
all_predicted_prices = []

# MJD Static Parameters
r_rate = 0.04
lam = 4.0
mu_J = -0.05
sigma_J = 0.15

contracts_tested = 0

for expire, strike in top_contracts:
    # Isolate the current contract
    contract_df = df_calls[(df_calls['EXPIRE_DATE'] == expire) &
                           (df_calls['STRIKE'] == strike)].copy()
    contract_df = contract_df.sort_values('QUOTE_DATE')

    # Merge with EGARCH forecast
    merged_df = pd.merge(contract_df, vol_forecast_df, on='QUOTE_DATE', how='inner')

    if len(merged_df) == 0:
        continue  # Skip if dates don't align

    contracts_tested += 1

    # Price every row in the current contract
    for index, row in merged_df.iterrows():
        S = row['UNDERLYING_LAST']
        K = row['STRIKE']
        T = row['DTE'] / 365.0
        sigma_continuous = row['EGARCH_VOL']

        mjd_price = merton_jump_call(S, K, T, r_rate, sigma_continuous, lam, mu_J, sigma_J)

        all_actual_prices.append(row['C_MID'])
        all_predicted_prices.append(mjd_price)

print(f"\nSuccessfully evaluated {len(all_actual_prices)} total data points across {contracts_tested} contracts.")

# ---------------------------------------------------------
# 6. Evaluating Aggregate Error and Scatter Plotting
# ---------------------------------------------------------
actuals = np.array(all_actual_prices)
predictions = np.array(all_predicted_prices)

safe_denominator = np.maximum(actuals, 1e-8)
percentage_errors = np.abs((predictions - actuals) / safe_denominator) * 100

mean_percentage_error = np.mean(percentage_errors)
mae = np.mean(np.abs(predictions - actuals))

print("\n--- Aggregate MJD + EGARCH Pricing Error ---")
print(f"Overall Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Overall Mean Absolute Percentage Error (MAPE): {mean_percentage_error:.2f}%")

# Scatter Plot Actual vs Predicted
plt.figure(figsize=(10, 8))

# Plot the ideal 1:1 pricing line (where prediction exactly equals actual)
max_val = max(np.max(actuals), np.max(predictions))
plt.plot([0, max_val], [0, max_val], color='black', linestyle='--', label='Perfect Pricing (y=x)')

# Plot our actual model predictions
plt.scatter(actuals, predictions, alpha=0.4, color='blue', edgecolor='k', label='EGARCH+MJD Forecasts')

plt.title('Aggregate Model Validation: Actual vs. Predicted Prices (Top 20 Contracts)', fontsize=14)
plt.xlabel('Actual Market Price (C_MID) $', fontsize=12)
plt.ylabel('Predicted Option Premium $', fontsize=12)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()