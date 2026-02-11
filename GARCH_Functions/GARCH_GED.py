import numpy as np
import pandas as pd
from arch import arch_model


def get_garch_ged_results(price_series):
    """
    Intakes raw price data, calculates log returns, and fits a GARCH(1,1)-GED model.

    Parameters:
    - price_series (pd.Series or np.array): Raw asset prices.

    Returns:
    - res (ARCHModelResult): The results object containing parameters,
      volatility, and residuals.
    """

    # 1. Calculate Log Returns: ln(Pt / Pt-1)
    # Dropping the first NaN value is required for the arch_model input
    log_returns = np.log(price_series / price_series.shift(1)).dropna()

    # 2. Scaling returns by 100
    # This is a standard practice in the 'arch' library to improve optimizer
    # convergence and prevent numerical errors.
    scaled_returns = 100 * log_returns

    # 3. Define and Fit GARCH(1,1) with GED
    # p=1, q=1 is the standard lag specification for volatility persistence.
    # 'ged' allows your team to analyze fat tails (leptokurtosis).
    model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='ged')

    # Suppressing iteration logs (disp='off') for a cleaner teammate experience
    res = model.fit(disp='off')

    return res