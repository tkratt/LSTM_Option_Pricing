import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. SETUP ---
file_name = "My_API_Data_UPDATED.xlsx"
image_folder = "Project_Graphs"

# Ensure folder exists
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

print(f"Reading data from {file_name}...")

# --- 2. LOAD DATA & CALCULATE RETURNS ---
try:
    df_prices = pd.read_excel(file_name, index_col=0)
except FileNotFoundError:
    print(f"ERROR: Could not find {file_name}. Is it in the folder?")
    exit()

# Log Returns
df_returns = np.log(df_prices / df_prices.shift(1)).dropna()

# --- 3. CALCULATE DYNAMIC CORRELATION (Rolling 60-Day Window) ---
# This mimics DCC by showing how the relationship evolves over time
window_size = 60 # 60 trading days = ~3 months
print(f"Calculating {window_size}-day rolling correlation between NVDA and TSLA...")

# Calculate Rolling Correlation specifically for NVDA and TSLA
rolling_corr = df_returns['NVDA'].rolling(window=window_size).corr(df_returns['TSLA'])

# --- 4. PLOT THE DECOUPLING ---
plt.figure(figsize=(12, 6))

# Plot the rolling correlation line
plt.plot(rolling_corr, color='#1f77b4', linewidth=2, label='Rolling Correlation (NVDA vs TSLA)')

# Add a horizontal red line at 0 (to show when they act oppositely)
plt.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7, label="Zero Correlation")

# Formatting
plt.title(f"Dynamic Correlation: Nvidia vs. Tesla (2012–2026)", fontsize=14, fontweight='bold')
plt.ylabel("Correlation Strength", fontsize=12)
plt.xlabel("Year", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# --- 5. SAVE THE PLOT ---
save_path = f"{image_folder}/NVDA_TSLA_Dynamic_Correlation.png"
plt.savefig(save_path, dpi=300) # High quality for PPT
plt.close()

print("="*50)
print(f"DONE! Correlation Graph saved to: {save_path}")
print("="*50)
print("Check your 'Project_Graphs' folder now.")