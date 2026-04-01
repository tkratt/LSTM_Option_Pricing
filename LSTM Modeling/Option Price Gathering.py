import pandas as pd
from pathlib import Path
import sys

# 1. UPDATE THIS PATH to where your Options DX folders are actually located!
base_path = Path("/Users/teddy/PycharmProjects/LSTM_Option_Pricing/Options_Price_Data")

# Quick safety check to ensure the folder exists
if not base_path.exists():
    print(f"Error: The folder {base_path} does not exist. Check your path!")
    sys.exit()

processed_chunks = []
file_count = 0

# 2. Loop through all .txt files
for file in base_path.rglob("*.txt"):
    print(f"Processing: {file.name}...")
    file_count += 1

    # Read the file
    df = pd.read_csv(file, skipinitialspace=True)

    # Clean column headers
    df.columns = df.columns.str.replace(r'\[|\]', '', regex=True).str.strip()

    # Filter for volume and DTE
    df = df[(df['C_VOLUME'] > 0) | (df['P_VOLUME'] > 0)]
    df = df[(df['DTE'] >= 30) & (df['DTE'] <= 90)]

    # Calculate mid prices
    df['C_MID'] = (df['C_BID'] + df['C_ASK']) / 2
    df['P_MID'] = (df['P_BID'] + df['P_ASK']) / 2

    processed_chunks.append(df)

# 3. Final safety check before combining
if file_count == 0:
    print("Error: No .txt files were found in that directory or its subfolders.")
    sys.exit()

print(f"\nSuccessfully processed {file_count} files. Combining data...")

# Combine and save
final_df = pd.concat(processed_chunks, ignore_index=True)
final_df.to_csv("Cleaned_SPX_Options.csv", index=False)
print("Done! Clean dataset saved.")