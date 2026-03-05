# Macro & Sector Features - Data Summary

## Overview
`macro_features.csv` has 5,030 daily observations from January 2006 to December 2025, pulled from Yahoo Finance and the FRED API. It has 21 features total that we're feeding into the Group 2 LSTM alongside the raw price/vol data that Pari pulled.

---

## Features & Rationale

### Volatility Indices (kept as levels)
- **VIX** - the standard fear index, gives us a sense of the overall market vol regime
- **OVX** - same idea but for crude oil implied vol
- **GVZ** - gold implied vol
- **VVIX** - vol of the VIX itself. When this spikes it means the market is uncertain about vol, which is exactly the kind of regime where our models will struggle most
- **SKEW** - measures demand for far OTM puts on the S&P. Captures tail risk that VIX alone doesn't pick up
- **MOVE** - bond market equivalent of VIX. Rate vol and equity vol are pretty connected, especially after 2022

These are all kept as levels since the actual value matters (VIX at 15 vs 40 is a fundamentally different environment).

### Rates & Spreads (kept as levels)
- **fed_funds_rate** - monetary policy regime, monthly from FRED and forward-filled to daily
- **hy_credit_spread** - high yield credit spread, good proxy for overall risk-off sentiment
- **yield_spread_10y2y** - 10Y minus 2Y Treasury yield, classic recession/rate environment signal
- **tips_5y_real_yield** - 5-year real interest rate from FRED. This one is specifically important for GLD and SLV since gold is very sensitive to real rates

### Sector & Asset Log Returns
- **XLK_logret** - tech sector ETF returns
- **XLE_logret** - energy sector ETF returns
- **QQQ_logret** - broad growth/risk appetite
- **SOXX_logret** - semiconductor ETF, relevant for Nvidia specifically
- **DXY_logret** - dollar index returns, a stronger dollar tends to pressure commodities
- **BTC_logret** - Bitcoin returns, mainly relevant for MSTR since it basically trades as leveraged BTC

These are converted to log returns since raw price levels are non-stationary.

### Earnings Dummies
Binary 0/1 flags for earnings announcement dates for META (53 days), TSLA (59), NVDA (78), AAPL (79), and MSTR (79). The individual tech stocks can move 5-15% on earnings days, so without flagging these the LSTM would just see a random vol spike with no explanation. Giving it a heads up that an earnings event happened should help.

---

## Date Constraints

Most features go back to January 2006 with no issues. The ones with limited history are:
- **OVX** starts around May 2007 (339 NaNs)
- **VVIX** starts around 2007 (260 NaNs)
- **GVZ** starts around June 2008 (607 NaNs)
- **BTC** only goes back to September 2014 (2,192 NaNs)

Everything else is either fully covered or has only a handful of scattered NaNs from FRED data gaps, which we forward-fill.

The practical implication is that if we want BTC as a feature for MSTR, our effective start date is late 2014. If we treat BTC as MSTR-specific and exclude it for other assets, we can go back to mid-2008 for everything else.