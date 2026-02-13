import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model

class GarchMeanResult:
    """
    A custom result class to mimic the standard ARCHModelResult.
    This allows the group to access .summary() and .forecast() easily.
    """
    def __init__(self, ols_model, garch_res, scaling_factor=100):
        self.ols_model = ols_model
        self.garch_res = garch_res
        self.scaling_factor = scaling_factor
        self._params = ols_model.params

    def summary(self):
        return self.ols_model.summary()

    def forecast(self, horizon=5):
        """
        Manually calculates GARCH-M forecasts.
        1. Forecasts Volatility (using GARCH).
        2. Forecasts Return (using the Risk Premium coefficient).
        """
        # 1. Forecast Volatility (Sigma)
        garch_forecast = self.garch_res.forecast(horizon=horizon)
        # Get the standard deviation (volatility) forecast
        pred_vol = np.sqrt(garch_forecast.variance.values[-1, :])
        
        # 2. Forecast Return
        # Formula: Return = Alpha + (Risk_Premium * Predicted_Volatility)
        alpha = self.ols_model.params['const']
        risk_premium = self.ols_model.params['x1']
        
        pred_return = alpha + (risk_premium * pred_vol)
        
        # Return a dictionary formatted like the standard arch library
        return {
            'h.1': horizon,
            'mean': pred_return / self.scaling_factor, # Scale back to decimals
            'variance': pred_vol**2
        }

def get_garch_mean_results(price_series):
    """
    Fits a GARCH-in-Mean model.
    
    METHODOLOGY:
    1. In-Sample: Fits GARCH to find volatility, then regresses Return on Volatility.
    2. Out-of-Sample: The returned object has a .forecast() method for predictions.
    
    Parameters:
    - price_series: Raw pricing data (e.g. df['Adj Close'])
    
    Returns:
    - result: A custom object with .summary() and .forecast() methods.
    """
    
    # 1. Prepare Data
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    scaling_factor = 100
    scaled_returns = log_returns * scaling_factor
    
    # 2. Stage 1: Fit Standard GARCH (Estimating Risk)
    # We use Normal distribution as requested.
    garch = arch_model(scaled_returns, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
    garch_res = garch.fit(disp='off')
    
    # 3. Stage 2: Fit In-Mean Regression (Estimating Risk Premium)
    conditional_volatility = garch_res.conditional_volatility
    
    Y = scaled_returns
    X = sm.add_constant(conditional_volatility) # Adds Intercept (Alpha)
    
    ols_model = sm.OLS(Y, X).fit()
    
    # 4. Return Custom Result Object
    return GarchMeanResult(ols_model, garch_res, scaling_factor)
