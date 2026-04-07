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
# HYBRID multi contract jump diff
import yfinance as yf
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ---------------------------------------------------------
# 1. Functions (FracDiff & Merton Jump Diffusion)
# ---------------------------------------------------------
def get_frac_diff_weights(d, length):
    weights = [1.0]
    for k in range(1, length):
        weights.append(-weights[-1] * (d - k + 1) / k)
    return np.array(weights)


def apply_frac_diff(series, d, window=40):
    weights = get_frac_diff_weights(d, window)[::-1]
    return series.rolling(window=window).apply(lambda x: np.dot(x, weights), raw=True)


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

# Dynamically find the top 20 contracts by row count
contract_counts = df_calls.groupby(['EXPIRE_DATE', 'STRIKE']).size().reset_index(name='Row_Count')
top_contracts_df = contract_counts.sort_values(by='Row_Count', ascending=False).head(20)

top_contracts = list(zip(top_contracts_df['EXPIRE_DATE'], top_contracts_df['STRIKE']))
print(f"Testing the top {len(top_contracts)} contracts...")

# ---------------------------------------------------------
# 3. Procure Underlying Data & Feature Engineering
# ---------------------------------------------------------
ticker = 'TSLA'
print(f"Fetching {ticker} data to train Hybrid Model...")
# Start early so the LSTM has years of training data before 2022
raw_data = yf.download(ticker, start='2012-01-01', end='2024-12-31')

df = pd.DataFrame()
df['Close'] = raw_data['Close']
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
df['Rolling_Vol'] = df['Log_Return'].rolling(window=22).std() * np.sqrt(252)
df['Log_Price'] = np.log(df['Close'])
df['FracDiff_Feature'] = apply_frac_diff(df['Log_Price'], d=0.7, window=40)

df = df.dropna()

# Explicitly start the test set right before your earliest option (e.g., April 2022)
test_start_date = '2022-01-01'
split_idx = df.index.get_loc(df[df.index >= test_start_date].index[0])

print("Generating EGARCH Conditional Volatility Features...")
am_train = arch_model(df['Log_Return'].iloc[:split_idx] * 100, vol='EGARCH', p=1, o=1, q=1, dist='Normal')
res_train = am_train.fit(disp='off')

am_full = arch_model(df['Log_Return'] * 100, vol='EGARCH', p=1, o=1, q=1, dist='Normal')
res_full = am_full.fix(res_train.params)

df['EGARCH_Vol'] = res_full.conditional_volatility / 100.0 * np.sqrt(252)
df['DayOfWeek'] = df.index.dayofweek
df['Month'] = df.index.month
df['DayOfYear'] = df.index.dayofyear

feature_cols = ['EGARCH_Vol', 'FracDiff_Feature', 'Log_Return', 'Rolling_Vol', 'DayOfWeek', 'Month', 'DayOfYear']
features = df[feature_cols].values
target_vol = df['Rolling_Vol'].values.reshape(-1, 1)

# ---------------------------------------------------------
# 4. Scaling & Walk-Forward Dataset Construction
# ---------------------------------------------------------
train_features, test_features = features[:split_idx], features[split_idx:]
train_target, test_target = target_vol[:split_idx], target_vol[split_idx:]

val_split_idx = int(len(train_features) * 0.90)

pure_train_features = train_features[:val_split_idx]
val_features = train_features[val_split_idx:]
pure_train_target = train_target[:val_split_idx]
val_target = train_target[val_split_idx:]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

pure_train_features_scaled = scaler_X.fit_transform(pure_train_features)
val_features_scaled = scaler_X.transform(val_features)
test_features_scaled = scaler_X.transform(test_features)

pure_train_target_scaled = scaler_y.fit_transform(pure_train_target)
val_target_scaled = scaler_y.transform(val_target)
test_target_scaled = scaler_y.transform(test_target)

lookback = 7
PRED_STEP = 1


def build_sequences(features_arr, target_arr, lookback=7):
    X, y = [], []
    for i in range(lookback, len(features_arr) - PRED_STEP + 1):
        X.append(features_arr[i - lookback: i])
        y.append(target_arr[i])
    return np.array(X), np.array(y)


X_train, y_train = build_sequences(pure_train_features_scaled, pure_train_target_scaled, lookback)
X_val, y_val = build_sequences(val_features_scaled, val_target_scaled, lookback)
X_test, y_test = build_sequences(test_features_scaled, test_target_scaled, lookback)

# ---------------------------------------------------------
# 5. Model Architecture & Training
# ---------------------------------------------------------
model = Sequential([
    LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dropout(0.2),
    Dense(1)
], name="Hybrid_EGARCH_LSTM")

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=Huber(), metrics=['mae'])
early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0)

print("Training Hybrid EGARCH-LSTM model...")
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=0
)

# ---------------------------------------------------------
# 6. Extraction & Date Alignment
# ---------------------------------------------------------
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred_raw = scaler_y.inverse_transform(y_pred_scaled)
y_pred = np.maximum(y_pred_raw, 1e-8)  # Enforce positive volatility

# Align dates: test set predictions start at `split_idx + lookback`
test_dates = df.index[split_idx + lookback:]

vol_forecast_df = pd.DataFrame({
    'QUOTE_DATE': test_dates.tz_localize(None),
    'HYBRID_VOL': y_pred.flatten()
})

# ---------------------------------------------------------
# 7. Multi-Contract Validation & Pricing
# ---------------------------------------------------------
print("\nValidating Hybrid Model against Real TSLA Option Prices...")

all_actual_prices = []
all_predicted_prices = []

# MJD Static Parameters (Identical to pure EGARCH test for fair comparison)
r_rate = 0.04
lam = 4.0
mu_J = -0.05
sigma_J = 0.15

contracts_tested = 0

for expire, strike in top_contracts:
    contract_df = df_calls[(df_calls['EXPIRE_DATE'] == expire) &
                           (df_calls['STRIKE'] == strike)].copy()
    contract_df = contract_df.sort_values('QUOTE_DATE')

    merged_df = pd.merge(contract_df, vol_forecast_df, on='QUOTE_DATE', how='inner')

    if len(merged_df) == 0:
        continue

    contracts_tested += 1

    for index, row in merged_df.iterrows():
        S = row['UNDERLYING_LAST']
        K = row['STRIKE']
        T = row['DTE'] / 365.0
        sigma_continuous = row['HYBRID_VOL']

        mjd_price = merton_jump_call(S, K, T, r_rate, sigma_continuous, lam, mu_J, sigma_J)

        all_actual_prices.append(row['C_MID'])
        all_predicted_prices.append(mjd_price)

print(f"\nSuccessfully evaluated {len(all_actual_prices)} total data points across {contracts_tested} contracts.")

# ---------------------------------------------------------
# 8. Evaluating Aggregate Error and Scatter Plotting
# ---------------------------------------------------------

actuals = np.array(all_actual_prices)
predictions = np.array(all_predicted_prices)

safe_denominator = np.maximum(actuals, 1e-8)
percentage_errors = np.abs((predictions - actuals) / safe_denominator) * 100

mean_percentage_error = np.mean(percentage_errors)
mae = np.mean(np.abs(predictions - actuals))

print("\n--- Aggregate Hybrid EGARCH-LSTM + MJD Pricing Error ---")
print(f"Overall Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Overall Mean Absolute Percentage Error (MAPE): {mean_percentage_error:.2f}%")

# Scatter Plot Actual vs Predicted
plt.figure(figsize=(10, 8))

# Plot the ideal 1:1 pricing line
max_val = max(np.max(actuals), np.max(predictions))
plt.plot([0, max_val], [0, max_val], color='black', linestyle='--', label='Perfect Pricing (y=x)')

# Plot actual model predictions
plt.scatter(actuals, predictions, alpha=0.4, color='purple', edgecolor='k', label='Hybrid+MJD Forecasts')

plt.title('Hybrid Model Validation: Actual vs. Predicted Prices (Top 20 Contracts)', fontsize=14)
plt.xlabel('Actual Market Price (C_MID) $', fontsize=12)
plt.ylabel('Predicted Option Premium $', fontsize=12)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()