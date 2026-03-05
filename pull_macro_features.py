import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "Data", "Assets", "macro_features.csv")

START = "2006-01-01"
END = "2025-12-31"
FRED_API_KEY = "API_KEY"

def fetch_fred(series_id):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "observation_start": START,
        "observation_end": END,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }
    r = requests.get(url, params=params)
    df = pd.DataFrame(r.json()["observations"])[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date").rename(columns={"value": series_id})

def fetch_yfinance(ticker, col_name):
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    return df[["Close"]].rename(columns={"Close": col_name})

def fetch_earnings_dummy(ticker, col_name, trading_days):
    t = yf.Ticker(ticker)
    dates = t.get_earnings_dates(limit=200)
    if dates is None or len(dates) == 0:
        return pd.Series(0, index=trading_days, name=col_name)
    earning_dates = pd.to_datetime(dates.index.date)
    dummy = pd.Series(0, index=trading_days, name=col_name)
    matched = trading_days.isin(earning_dates)
    dummy[matched] = 1
    return dummy

# --- yfinance pulls ---
vix  = fetch_yfinance("^VIX",     "VIX")
dxy  = fetch_yfinance("DX-Y.NYB", "DXY")
xlk  = fetch_yfinance("XLK",      "XLK_close")
xle  = fetch_yfinance("XLE",      "XLE_close")
qqq  = fetch_yfinance("QQQ",      "QQQ_close")
btc  = fetch_yfinance("BTC-USD",  "BTC_close")
ovx  = fetch_yfinance("^OVX",     "OVX")
gvz  = fetch_yfinance("^GVZ",     "GVZ")
soxx = fetch_yfinance("SOXX",     "SOXX_close")
vvix = fetch_yfinance("^VVIX",    "VVIX")
skew = fetch_yfinance("^SKEW",    "SKEW")
move = fetch_yfinance("^MOVE",    "MOVE")

# --- FRED pulls ---
dgs10     = fetch_fred("DGS10")
dgs2      = fetch_fred("DGS2")
fedfunds  = fetch_fred("FEDFUNDS")
hy_spread = fetch_fred("BAMLH0A0HYM2")
tips5y    = fetch_fred("DFII5")       # 5Y TIPS real yield — key for GLD/SLV

# --- Merge yfinance data ---
yf_data = vix.join([dxy, xlk, xle, qqq, btc, ovx, gvz, soxx, vvix, skew, move], how="outer")
yf_data.index.name = "date"

# --- Merge FRED data ---
fred_data = dgs10.join([dgs2, fedfunds, hy_spread, tips5y], how="outer")
fred_data["yield_spread_10y2y"] = fred_data["DGS10"] - fred_data["DGS2"]
fred_data = fred_data.drop(columns=["DGS10", "DGS2"])
fred_data.columns = ["fed_funds_rate", "hy_credit_spread", "tips_5y_real_yield", "yield_spread_10y2y"]

df = yf_data.join(fred_data, how="outer")

# Drop weekends/holidays where all yfinance data is NaN
df = df.dropna(subset=["VIX"])

# Forward fill rate/spread series
df["fed_funds_rate"]     = df["fed_funds_rate"].ffill()
df["tips_5y_real_yield"] = df["tips_5y_real_yield"].ffill()
df[["hy_credit_spread"]] = df[["hy_credit_spread"]].ffill(limit=5)

# Price-based series: convert to log returns (stationary)
price_cols = ["XLK_close", "XLE_close", "QQQ_close", "SOXX_close", "DXY", "BTC_close"]
for col in price_cols:
    new_name = col.replace("_close", "_logret") if "_close" in col else col + "_logret"
    df[new_name] = np.log(df[col]).diff()
df = df.drop(columns=price_cols)

# --- Earnings dummies ---
trading_days = df.index
earnings_tickers = {
    "META":  "meta_earnings",
    "TSLA":  "tsla_earnings",
    "NVDA":  "nvda_earnings",
    "AAPL":  "aapl_earnings",
    "MSTR":  "mstr_earnings",
}
for ticker, col_name in earnings_tickers.items():
    df[col_name] = fetch_earnings_dummy(ticker, col_name, trading_days)

df.index.name = "date"
df.to_csv(OUTPUT_PATH)
print(f"Saved {OUTPUT_PATH} — {len(df)} rows, {len(df.columns)} columns")
print(df.tail())