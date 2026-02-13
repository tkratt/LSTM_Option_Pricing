import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")


# Diebold–Mariano test with simple HAC adjustment
def diebold_mariano_test(loss1, loss2, h=1):
    d = loss1 - loss2
    T = len(d)
    mean_d = np.mean(d)

    gamma0 = np.var(d, ddof=1)
    var_d = gamma0

    for lag in range(1, h):
        gamma = np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
        var_d += 2 * gamma

    dm_stat = mean_d / np.sqrt(var_d / T)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


# Simple bootstrap Model Confidence Set
def model_confidence_set(loss_dict, B=500, alpha=0.05):
    models = list(loss_dict.keys())
    losses = np.column_stack([loss_dict[m] for m in models])
    T = losses.shape[0]

    mean_losses = losses.mean(axis=0)
    max_stats = []

    for _ in range(B):
        idx = np.random.randint(0, T, T)
        boot_losses = losses[idx]
        boot_means = boot_losses.mean(axis=0)
        max_stats.append(np.max(boot_means - boot_means.min()))

    critical_value = np.percentile(max_stats, 100 * (1 - alpha))

    superior_models = []
    for i, m in enumerate(models):
        diff = mean_losses[i] - mean_losses.min()
        if diff <= critical_value:
            superior_models.append(m)

    return superior_models


# Download data
ticker = "^GSPC"
data = yf.download(ticker, start="2006-01-01", end="2025-12-31", progress=False)

returns = 100 * data["Close"].pct_change().dropna()

test_len = 500
train_len = len(returns) - test_len
realized_vol = returns ** 2


# Model specifications
model_specs = {
    "GARCH": {"vol": "GARCH", "p": 1, "q": 1, "dist": "normal"},
    "GARCH-t": {"vol": "GARCH", "p": 1, "q": 1, "dist": "t"},
    "EGARCH": {"vol": "EGARCH", "p": 1, "q": 1, "dist": "t"},
    "GJR-GARCH": {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "t"},
    "FIGARCH": {"vol": "FIGARCH", "p": 1, "q": 1, "dist": "t"},
}


all_loss_series = {}
results = []

for name, spec in model_specs.items():
    print(f"Estimating {name}...")

    # Fit once on initial training sample (for BIC and diagnostics)
    model_full = arch_model(returns.iloc[:train_len], mean="Zero", **spec)
    res_full = model_full.fit(disp="off")

    bic = res_full.bic

    std_resid_sq = (res_full.resid / res_full.conditional_volatility) ** 2
    lb_pvalue = acorr_ljungbox(std_resid_sq.dropna(), lags=[10])["lb_pvalue"].iloc[0]

    forecasts = []

    # Expanding window forecasts
    for t in range(train_len, len(returns)):
        train_data = returns.iloc[:t]

        try:
            model = arch_model(train_data, mean="Zero", **spec)
            res = model.fit(disp="off")
            fcast = res.forecast(horizon=1)
            forecasts.append(fcast.variance.iloc[-1, 0])
        except:
            forecasts.append(np.nan)

    forecasts = np.array(forecasts)
    actual = realized_vol.iloc[train_len:].values

    mask = ~np.isnan(forecasts)
    forecasts = forecasts[mask]
    actual = actual[mask]

    # QLIKE loss
    ratio = actual / forecasts
    qlike = ratio - np.log(ratio) - 1

    # MSE
    mse = np.mean((forecasts - actual) ** 2)

    all_loss_series[name] = qlike

    results.append({
        "Model": name,
        "QLIKE": np.mean(qlike),
        "MSE": mse,
        "BIC": bic,
        "LB_pVal": lb_pvalue
    })


# Summary table
results_df = pd.DataFrame(results).sort_values("QLIKE")
print("\nModel Comparison")
print(results_df.to_string(index=False))


# DM test for top two models
top = results_df.iloc[0]["Model"]
second = results_df.iloc[1]["Model"]

dm_stat, dm_p = diebold_mariano_test(
    all_loss_series[top],
    all_loss_series[second]
)

print("\nDiebold–Mariano Test")
print(f"{top} vs {second}")
print(f"DM Statistic: {dm_stat:.4f} | p-value: {dm_p:.4f}")


# Model Confidence Set
mcs_models = model_confidence_set(all_loss_series, B=500, alpha=0.05)

print("\nModel Confidence Set (95%)")
print("Superior models:", mcs_models)
