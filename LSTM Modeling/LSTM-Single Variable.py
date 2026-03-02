import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Data Procurement & Feature Engineering
# ---------------------------------------------------------
# Fetch S&P 500 data from 2006 to 2025
print("Fetching data...")
df = yf.download('^GSPC', start='2006-01-01', end='2025-12-31')

# Calculate daily log returns
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

# Target Variable: 1-step ahead proxy for variance (squared log return)
df['Variance_Proxy'] = df['Log_Return'] ** 2

# Drop the initial NaN from shifting
df = df.dropna()

# Extract the numpy arrays for scaling
returns = df['Log_Return'].values.reshape(-1, 1)
variance = df['Variance_Proxy'].values.reshape(-1, 1)

# ---------------------------------------------------------
# 2. Scaling & 3D Tensor Formatting
# ---------------------------------------------------------
# We scale features and targets separately so we can easily inverse-transform predictions later
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

returns_scaled = scaler_X.fit_transform(returns)
variance_scaled = scaler_y.fit_transform(variance)

lookback = 21
X, y = [], []

# Create the 3D tensor: predicting variance at day 'i' using returns from 'i-21' to 'i-1'
for i in range(lookback, len(returns_scaled)):
    X.append(returns_scaled[i-lookback:i])
    y.append(variance_scaled[i])

X = np.array(X)
y = np.array(y)

# Chronological Train/Test Split (80% Train, 20% Test)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"X_train shape: {X_train.shape}") # Should be (Samples, 21, 1)

# ---------------------------------------------------------
# 3. Model Architecture & Training
# ---------------------------------------------------------
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))

# Standardize on Huber Loss, tracking MAE as a metric, with lr=0.001
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=Huber(), metrics=['mae'])

print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1, # Uses 10% of training data for validation
    verbose=1
)

# ---------------------------------------------------------
# 4. Evaluation & Percentage Metrics
# ---------------------------------------------------------
y_pred_scaled = model.predict(X_test)

# Inverse transform to get actual variance numbers back
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# 1. Standard Metrics
mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred)

# 2. Percentage Metric (MAPE)
# Multiplied by 100 to format as a standard percentage (e.g., 45.5%)
mape = mean_absolute_percentage_error(y_test_actual, y_pred) * 100

# 3. Custom Metric (QLIKE)
def calculate_qlike(actual, predicted):
    epsilon = 1e-8
    actual = np.maximum(actual, epsilon)
    predicted = np.maximum(predicted, epsilon)
    qlike_values = (actual / predicted) - np.log(actual / predicted) - 1
    return np.mean(qlike_values)

qlike = calculate_qlike(y_test_actual, y_pred)

# 4. Annualized Volatility Conversion (For Option Pricing Context)
# Add a floor of 0 to ensure no negative numbers are passed to np.sqrt()
y_pred_safe = np.maximum(y_pred, 0)
y_test_actual_safe = np.maximum(y_test_actual, 0)

# Convert actual and predicted variance arrays into Annualized Volatility %
actual_ann_vol_pct = np.sqrt(y_test_actual_safe) * np.sqrt(252) * 100
pred_ann_vol_pct = np.sqrt(y_pred_safe) * np.sqrt(252) * 100

# Calculate how far off the model is in Annualized Volatility terms
ann_vol_mae = mean_absolute_error(actual_ann_vol_pct, pred_ann_vol_pct)
print("\n--- Final Evaluation Metrics ---")
print(f"MSE:   {mse:.7f}")
print(f"RMSE:  {rmse:.7f}")
print(f"MAE:   {mae:.7f}")
print(f"QLIKE: {qlike:.2f}")
print("\n--- Percentage Metrics (Easier to Understand) ---")
print(f"Annualized Volatility MAE: {ann_vol_mae:.2f}% (How far off the % volatility is)")

# ---------------------------------------------------------
# 5. Visualization: Actual vs. Predicted Variance
# ---------------------------------------------------------
# Set the plot size for better readability
plt.figure(figsize=(14, 6))

# Plot the actual realized variance
plt.plot(y_test_actual, label='Actual 1-Step Variance', color='black', linewidth=1.5, alpha=0.7)

# Plot the LSTM's predictions
plt.plot(y_pred, label='LSTM Predicted Variance', color='red', linewidth=1.5, alpha=0.8)

# Formatting the chart
plt.title('S&P 500: Actual vs. LSTM Predicted Variance (Test Set)', fontsize=14)
plt.xlabel('Trading Days (Test Period)', fontsize=12)
plt.ylabel('Variance (Squared Log Returns)', fontsize=12)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Tight layout keeps the labels from getting cut off
plt.tight_layout()
plt.show()