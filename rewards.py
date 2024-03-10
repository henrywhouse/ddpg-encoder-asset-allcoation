###############
### Rewards ###
###############

import numpy as np

def rolling_sharpe_ratio(returns, rf):
    """Calculates rolling Sharpe ratio over holding period and returns it"""

    excess_returns = returns - (rf / 252)  # Adjust rf_annual for daily compounding
    expected_excess =  np.mean(excess_returns)  # Calculate expected (average) return
    sigma = np.std(excess_returns)  # ddof=1 for sample std dev
    rolling_sharpe = expected_excess / sigma * np.sqrt(252)  # Adjust for annualization
    return rolling_sharpe


def rolling_market_ratio(returns, rf, benchmark):
    """Calculates rolling Sharpe ratio with a benchmark over a holding period and returns it"""

    excess_returns = returns - (rf / 252)  # Adjust rf for daily compounding
    expected_excess =  np.mean(excess_returns)  # Calculate expected (average) return
    covariance = np.cov(np.squeeze(excess_returns), np.squeeze(benchmark))[0,1]  # Calculate covariance with benchmark (market) portfolio
    var_rm = np.var(benchmark)  # ddof=1 for sample std dev
    beta = covariance / var_rm
    rolling_market_sharpe = expected_excess / beta * np.sqrt(252)  # Adjust for annualization
    return rolling_market_sharpe


def rolling_alpha(returns, rf, benchmark):
    """Calculates rolling Sharpe ratio with a benchmark over a holding period and returns it"""

    rf = rf / 252 # Adjust rf for daily compounding
    covariance = np.cov(np.squeeze(returns), np.squeeze(benchmark), ddof=1)[0,1]  # Calculate covariance with benchmark (market) portfolio
    var_rm = np.var(benchmark, ddof=1)
    beta = covariance / var_rm
    alpha = np.mean(returns) - np.mean(rf) - beta * (np.mean(benchmark) - np.mean(rf))
    annualized_alpha = alpha * np.sqrt(252)  # Annualization
    return annualized_alpha