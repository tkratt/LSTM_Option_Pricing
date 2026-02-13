from arch import arch_model
from arch.univariate.base import ARCHModelResult
import pandas as pd


def fit_figarch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = 't',
    mean: str = 'Zero',
    scale: float = 100,
) -> ARCHModelResult:
    """
    Fit a FIGARCH(p, d, q) model.
    
    Parameters
    ----------
    returns : pd.Series
        Log return series.
    p : int
        GARCH order.
    q : int
        ARCH order.
    dist : str
        Error distribution: 'normal', 't', 'skewt', 'ged'.
    mean : str
        Mean model: 'Zero', 'Constant', 'AR'.
    scale : float
        Multiply returns by this (100 = percentage returns).
    
    Returns
    -------
    ARCHModelResult
        Fitted model result. Call .summary() for details.
    
    Examples
    --------
    >>> res = fit_figarch(prep.returns, p=1, q=1, dist='t')
    >>> print(res.summary())
    >>> print(f"d = {res.params['d']:.4f}")  # long memory parameter
    """
    am = arch_model(
        returns * scale,
        mean=mean,
        vol='FIGARCH',
        p=p,
        q=q,
        dist=dist,
    )
    return am.fit(disp='off')
