import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
import os

# --- 1. SETUP ---
file_name = "My_API_Data_UPDATED.xlsx"

# Create a folder for the images so they don't clutter your desktop
image_folder = "Project_Graphs"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

print(f"Reading data from {file_name}...")

# --- 2. LOAD & CALCULATE ---
df_prices = pd.read_excel(file_name, index_col=0)
df_returns = np.log(df_prices / df_prices.shift(1)).dropna()

# --- 3. LOOP & SAVE IMAGES ---
for ticker in df_returns.columns:
    print(f"\nProcessing {ticker}...")
    r_t = df_returns[ticker]

    # A. Stationarity Test
    result = adfuller(r_t)
    print(f"  > ADF p-value: {result[1]:.8f}")

    # B. Volatility Clustering (Save as Image)
    plt.figure(figsize=(10, 4))
    plt.plot(r_t**2, color='orange', alpha=0.8)
    plt.title(f"{ticker}: Volatility Clustering")
    plt.ylabel("Squared Returns")
    plt.grid(True, alpha=0.3)
    
    # SAVE instead of show
    save_path = f"{image_folder}/{ticker}_Volatility.png"
    plt.savefig(save_path)
    plt.close() # Close memory to keep computer fast
    print(f"  > Saved graph to: {save_path}")

    # C. Q-Q Plot (Save as Image)
    plt.figure(figsize=(6, 6))
    stats.probplot(r_t, dist="norm", plot=plt)
    plt.title(f"{ticker}: Q-Q Plot")
    
    # SAVE instead of show
    save_path_qq = f"{image_folder}/{ticker}_QQ_Plot.png"
    plt.savefig(save_path_qq)
    plt.close()
    print(f"  > Saved graph to: {save_path_qq}")

print(f"\nDONE! All graphs are in the '{image_folder}' folder.")
