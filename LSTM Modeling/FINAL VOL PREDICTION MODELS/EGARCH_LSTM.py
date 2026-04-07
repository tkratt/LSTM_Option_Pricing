#EGARCH LSTM
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from arch import arch_model

# ---------------------------------------------------------
# 1. Data Procurement & Feature Engineering
# ---------------------------------------------------------
print("Fetching data...")
df = yf.download('^GSPC', start='2006-01-01', end='2025-12-31')

# Flatten yfinance's new MultiIndex columns if they exist
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Calculate standard daily log returns
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

# Drop the first NaN so the ARCH model doesn't fail
df = df.dropna(subset=['Log_Return'])

# --- EGARCH Conditional Variance ---
print("Fitting EGARCH model...")
scaled_returns = df['Log_Return'] * 100
egarch_model = arch_model(scaled_returns, vol='EGARCH', p=1, o=1, q=1, dist='normal')
egarch_results = egarch_model.fit(update_freq=0, disp='off')

# Extract conditional volatility, square it, and un-scale it
df['EGARCH_Variance'] = (egarch_results.conditional_volatility ** 2) / 10000

# Target Variable & Input Feature: Proxy for realized variance
df['Variance_Proxy'] = df['Log_Return'] ** 2

# Drop any remaining NaNs
df = df.dropna()

# --- NEW: ONLY VARIANCE AND EGARCH AS INPUTS ---
feature_cols = ['Variance_Proxy', 'EGARCH_Variance']
features = df[feature_cols].values

# The target we want to predict
variance = df['Variance_Proxy'].values.reshape(-1, 1)

# ---------------------------------------------------------
# 2. Scaling & 3D Tensor Formatting
# ---------------------------------------------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Scales the 2 columns in the features array
features_scaled = scaler_X.fit_transform(features)
variance_scaled = scaler_y.fit_transform(variance)

lookback = 21
X, y = [], []

for i in range(lookback, len(features_scaled)):
    # Grabs the previous 21 days for the 2 features
    X.append(features_scaled[i-lookback:i])
    y.append(variance_scaled[i])

X = np.array(X)
y = np.array(y)

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Will now output (Samples, 21, 2)
print(f"X_train shape: {X_train.shape}")

# ---------------------------------------------------------
# 3. Model Architecture & Training
# ---------------------------------------------------------
model = Sequential()
# Input shape adapts to 2 features
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))

# Softplus ensures variance predictions never dip below 0
model.add(Dense(1, activation='softplus'))

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=Huber(), metrics=['mae'])

print("Training LSTM model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)

# ---------------------------------------------------------
# 4. Evaluation & Percentage Metrics
# ---------------------------------------------------------
y_pred_scaled = model.predict(X_test)

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred)
mape = mean_absolute_percentage_error(y_test_actual, y_pred) * 100

def calculate_qlike(actual, predicted):
    epsilon = 1e-8
    actual = np.maximum(actual, epsilon)
    predicted = np.maximum(predicted, epsilon)
    qlike_values = (actual / predicted) - np.log(actual / predicted) - 1
    return np.mean(qlike_values)

qlike = calculate_qlike(y_test_actual, y_pred)

y_pred_safe = np.maximum(y_pred, 0)
y_test_actual_safe = np.maximum(y_test_actual, 0)

actual_ann_vol_pct = np.sqrt(y_test_actual_safe) * np.sqrt(252) * 100
pred_ann_vol_pct = np.sqrt(y_pred_safe) * np.sqrt(252) * 100

ann_vol_mae = mean_absolute_error(actual_ann_vol_pct, pred_ann_vol_pct)

print("\n--- Final Evaluation Metrics ---")
print(f"MSE:   {mse:.7f}")
print(f"RMSE:  {rmse:.7f}")
print(f"MAE:   {mae:.7f}")
print(f"QLIKE: {qlike:.2f}")
print("\n--- Percentage Metrics ---")
print(f"Annualized Volatility MAE: {ann_vol_mae:.2f}%")

# ---------------------------------------------------------
# 5. Visualization: Actual vs. Predicted Variance
# ---------------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, label='Actual Realized Variance', color='black', linewidth=1.5, alpha=0.7)
plt.plot(y_pred, label='LSTM Predicted Variance', color='red', linewidth=1.5, alpha=0.8)

plt.title('S&P 500: Actual vs. LSTM Predicted Variance (EGARCH)', fontsize=14)
plt.xlabel('Trading Days (Test Period)', fontsize=12)
plt.ylabel('Variance (Squared Log Returns)', fontsize=12)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()