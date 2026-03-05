#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
"""
Modifications:
Scale before splitting
Fill NaN macro data with previous available data, NaN return data with 0
Annulized vol% calculated without multiplying by 100
Softplus for output layer
"""
# ---------------------------------------------------------
# Load and Clean Data
# ---------------------------------------------------------
file_path = "/Users/zhangluolin/Downloads/group_vol&macrodata.csv"
df = pd.read_csv(file_path)

df = df.sort_values("Date").reset_index(drop=True)

# No forward fill; just drop missing rows
df = df.dropna().reset_index(drop=True)

# ---------------------------------------------------------
# Define Feature Columns and Target Columns
# ---------------------------------------------------------
asset_log_returns = [
    "GSPC_Log_Ret","GLD_Log_Ret","SLV_Log_Ret","CL_Log_Ret",
    "META_Log_Ret","TSLA_Log_Ret","NVDA_Log_Ret","AAPL_Log_Ret","MSTR_Log_Ret"
]

asset_variance = [
    "GSPC_Variance","GLD_Variance","SLV_Variance","CL_Variance",
    "META_Variance","TSLA_Variance","NVDA_Variance","AAPL_Variance","MSTR_Variance"
]


macro_cols = [
    "VIX","OVX","GVZ","VVIX","SKEW","MOVE",
    "fed_funds_rate","hy_credit_spread",
    "tips_5y_real_yield","yield_spread_10y2y"
]

df[macro_cols] = df[macro_cols].ffill()

# sector / crypto returns
ret_cols = ["XLK_logret","XLE_logret","QQQ_logret","SOXX_logret","DXY_logret","BTC_logret"]
df[ret_cols] = df[ret_cols].fillna(0)

# finally drop rows that still have missing values
df = df.dropna().reset_index(drop=True)
feature_columns = asset_log_returns + asset_variance + macro_cols + ret_cols  # 39 cols

target_columns = [
    "GSPC_Target_Variance_t+1","GLD_Target_Variance_t+1","SLV_Target_Variance_t+1",
    "CL_Target_Variance_t+1","META_Target_Variance_t+1","TSLA_Target_Variance_t+1",
    "NVDA_Target_Variance_t+1","AAPL_Target_Variance_t+1","MSTR_Target_Variance_t+1"
]

# ---------------------------------------------------------
# Extract X_raw and y_raw
# ---------------------------------------------------------
X_raw = df[feature_columns].astype(np.float32).values  # (T, 39)
y_raw = df[target_columns].astype(np.float32).values   # (T, 9)

# ---------------------------------------------------------
# Set Lookback
# ---------------------------------------------------------
lookback = 21
series_names = ["GSPC","GLD","SLV","CL","META","TSLA","NVDA","AAPL","MSTR"]

results = []
all_actual = []
all_pred = []

for j, name in enumerate(series_names):
    print(f"\n==================== {name} ====================")

    # --- ONE target series (no log transform) ---
    y1_raw = y_raw[:, j].reshape(-1, 1)

    # --- build sequences ---
    X_seq, y_seq = [], []
    for i in range(lookback, len(X_raw)):
        X_seq.append(X_raw[i-lookback:i, :])
        y_seq.append(y1_raw[i, :])   

    X_seq = np.array(X_seq)  
    y_seq = np.array(y_seq)  

    # --- split ---
    split_idx = int(len(X_seq) * 0.8)
    X_train_raw, X_test_raw = X_seq[:split_idx], X_seq[split_idx:]
    y_train_raw, y_test_raw = y_seq[:split_idx], y_seq[split_idx:]

    # --- scale ---
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_2d = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    X_test_2d  = X_test_raw.reshape(-1, X_test_raw.shape[-1])

    scaler_X.fit(X_train_2d)
    X_train = scaler_X.transform(X_train_2d).reshape(X_train_raw.shape)
    X_test  = scaler_X.transform(X_test_2d).reshape(X_test_raw.shape)

    scaler_y.fit(y_train_raw)
    y_train = scaler_y.transform(y_train_raw)
    y_test  = scaler_y.transform(y_test_raw)

    # --- chronological validation split ---
    val_frac = 0.1
    val_idx = int(len(X_train) * (1 - val_frac))
    X_tr, X_val = X_train[:val_idx], X_train[val_idx:]
    y_tr, y_val = y_train[:val_idx], y_train[val_idx:]

    # --- model ---
    model = Sequential()
    model.add(Input(shape=(X_tr.shape[1], X_tr.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1, activation="softplus"))

    model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(), metrics=['mae'])

    model.fit(
        X_tr, y_tr,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        shuffle=False,
        verbose=0
    )

    # --- predict and inverse ---
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_actual = y_test_raw

    all_actual.append(y_test_actual.flatten())
    all_pred.append(y_pred.flatten())

    # --- metrics ---
    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_pred)

    # QLIKE 
    def calculate_qlike(actual, predicted):
        eps = 1e-8
        actual = np.maximum(actual, eps)
        predicted = np.maximum(predicted, eps)
        qlike_values = (actual / predicted) - np.log(actual / predicted) - 1
        return np.mean(qlike_values)

    qlike = calculate_qlike(y_test_actual, y_pred)

    y_pred_safe = np.maximum(y_pred, 0.0)
    y_test_safe = np.maximum(y_test_actual, 0.0)

    actual_ann_vol_pct = np.sqrt(y_test_safe) * np.sqrt(252)
    pred_ann_vol_pct   = np.sqrt(y_pred_safe) * np.sqrt(252)

    ann_vol_mae = mean_absolute_error(actual_ann_vol_pct, pred_ann_vol_pct)

    print(
        f"MSE: {mse:.7f} | RMSE: {rmse:.7f} | MAE: {mae:.7f} | "
        f"QLIKE: {qlike:.4f} | AnnVol MAE: {ann_vol_mae:.2f}%"
    )

    results.append([name, mse, rmse, mae, qlike, ann_vol_mae])

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
results_df = pd.DataFrame(results, columns=["Asset","MSE","RMSE","MAE","QLIKE","AnnVol_MAE(%)"])
print("\n\n=== Summary ===")
print(results_df)

# ---------------------------------------------------------
# Plots 
# ---------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.plot(all_actual[i], color='black', linewidth=1.5, alpha=0.7, label="Actual Variance")
    ax.plot(all_pred[i], color='red', linewidth=1.5, alpha=0.8, label="Predicted Variance")

    ax.set_title(series_names[i], fontsize=12)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Variance")
    ax.grid(True, linestyle='--', alpha=0.5)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")

plt.suptitle("Actual vs LSTM Predicted Variance (Test Set)", fontsize=16)
plt.tight_layout()
plt.show()


# In[ ]:




