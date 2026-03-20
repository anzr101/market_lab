import numpy as np
import logging
from pathlib import Path
from typing import Dict


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "metrics.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def sharpe_ratio(returns: np.ndarray) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Computes the ratio of mean return to standard deviation, scaled to annual
    terms assuming 252 trading days per year.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of period returns (daily returns assumed).
        
    Returns
    -------
    float
        Annualized Sharpe ratio. Returns 0.0 if standard deviation is zero.
        
    Raises
    ------
    ValueError
        If returns array is empty or contains non-finite values.
    """
    logger.info("Computing Sharpe ratio")
    
    if len(returns) == 0:
        logger.error("Empty returns array provided")
        raise ValueError("Returns array cannot be empty")
    
    if not np.all(np.isfinite(returns)):
        logger.error("Returns contain non-finite values (NaN or Inf)")
        raise ValueError("Returns must contain only finite values")
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    if std_return == 0 or np.isnan(std_return):
        logger.warning("Standard deviation is zero or NaN, returning Sharpe ratio = 0")
        return 0.0
    
    sharpe = (mean_return / std_return) * np.sqrt(252)
    
    logger.info(f"Sharpe ratio: {sharpe:.4f}")
    
    return float(sharpe)


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown from equity curve.
    
    Computes the largest peak-to-trough decline in the equity curve,
    expressed as a percentage of the peak value.
    
    Parameters
    ----------
    equity_curve : np.ndarray
        Array of cumulative equity values over time.
        
    Returns
    -------
    float
        Maximum drawdown as a decimal (e.g., -0.25 for 25% drawdown).
        Returns 0.0 if equity curve is monotonically increasing.
        
    Raises
    ------
    ValueError
        If equity_curve is empty or contains non-finite values.
    """
    logger.info("Computing maximum drawdown")
    
    if len(equity_curve) == 0:
        logger.error("Empty equity curve provided")
        raise ValueError("Equity curve cannot be empty")
    
    if not np.all(np.isfinite(equity_curve)):
        logger.error("Equity curve contains non-finite values (NaN or Inf)")
        raise ValueError("Equity curve must contain only finite values")
    
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = np.min(drawdowns)
    
    logger.info(f"Maximum drawdown: {max_dd:.6f} ({max_dd*100:.2f}%)")
    
    return float(max_dd)


def win_rate(returns: np.ndarray) -> float:
    """
    Calculate win rate (percentage of positive returns).
    
    Computes the fraction of returns that are strictly positive.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of period returns.
        
    Returns
    -------
    float
        Win rate as a decimal between 0 and 1 (e.g., 0.55 for 55% win rate).
        
    Raises
    ------
    ValueError
        If returns array is empty or contains non-finite values.
    """
    logger.info("Computing win rate")
    
    if len(returns) == 0:
        logger.error("Empty returns array provided")
        raise ValueError("Returns array cannot be empty")
    
    if not np.all(np.isfinite(returns)):
        logger.error("Returns contain non-finite values (NaN or Inf)")
        raise ValueError("Returns must contain only finite values")
    
    winning_trades = np.sum(returns > 0)
    total_trades = len(returns)
    win_rate_value = winning_trades / total_trades
    
    logger.info(f"Win rate: {win_rate_value:.4f} ({win_rate_value*100:.2f}%)")
    logger.info(f"Winning trades: {winning_trades}/{total_trades}")
    
    return float(win_rate_value)


def profit_factor(returns: np.ndarray) -> float:
    """
    Calculate profit factor (ratio of gross profits to gross losses).
    
    Computes the sum of all positive returns divided by the absolute value
    of the sum of all negative returns. A profit factor > 1 indicates
    positive expected returns.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of period returns.
        
    Returns
    -------
    float
        Profit factor. Returns np.inf if there are no losses.
        Returns 0.0 if there are no profits.
        
    Raises
    ------
    ValueError
        If returns array is empty or contains non-finite values.
    """
    logger.info("Computing profit factor")
    
    if len(returns) == 0:
        logger.error("Empty returns array provided")
        raise ValueError("Returns array cannot be empty")
    
    if not np.all(np.isfinite(returns)):
        logger.error("Returns contain non-finite values (NaN or Inf)")
        raise ValueError("Returns must contain only finite values")
    
    profits = returns[returns > 0]
    losses = returns[returns < 0]
    
    total_profits = np.sum(profits) if len(profits) > 0 else 0.0
    total_losses = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0
    
    if total_losses == 0:
        if total_profits > 0:
            logger.warning("No losses detected, profit factor = inf")
            return np.inf
        else:
            logger.warning("No profits and no losses, profit factor = 0")
            return 0.0
    
    pf = total_profits / total_losses
    
    logger.info(f"Profit factor: {pf:.4f}")
    logger.info(f"Total profits: {total_profits:.6f}, Total losses: {total_losses:.6f}")
    
    return float(pf)


def summary_stats(returns: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive summary statistics for returns series.
    
    Computes equity curve from returns and calculates all key performance
    metrics including mean, standard deviation, Sharpe ratio, maximum drawdown,
    win rate, and profit factor.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of period returns (daily returns assumed).
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - mean: average return
        - std: standard deviation of returns
        - sharpe: annualized Sharpe ratio
        - max_drawdown: maximum drawdown percentage
        - win_rate: fraction of positive returns
        - profit_factor: ratio of gross profits to gross losses
        
    Raises
    ------
    ValueError
        If returns array is empty or contains non-finite values.
        
    Notes
    -----
    Equity curve computed as cumulative product of (1 + returns).
    All metrics assume daily returns for annualization purposes.
    """
    logger.info("="*60)
    logger.info("COMPUTING SUMMARY STATISTICS")
    logger.info("="*60)
    
    if len(returns) == 0:
        logger.error("Empty returns array provided")
        raise ValueError("Returns array cannot be empty")
    
    if not np.all(np.isfinite(returns)):
        logger.error("Returns contain non-finite values (NaN or Inf)")
        raise ValueError("Returns must contain only finite values")
    
    logger.info(f"Returns array length: {len(returns)}")
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    logger.info(f"Mean return: {mean_return:.6f}")
    logger.info(f"Std deviation: {std_return:.6f}")
    
    sharpe = sharpe_ratio(returns)
    
    equity_curve = np.cumprod(1 + returns)
    max_dd = max_drawdown(equity_curve)
    
    wr = win_rate(returns)
    
    pf = profit_factor(returns)
    
    stats = {
        'mean': float(mean_return),
        'std': float(std_return),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'win_rate': float(wr),
        'profit_factor': float(pf)
    }
    
    logger.info("="*60)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*60)
    for key, value in stats.items():
        logger.info(f"{key}: {value:.6f}")
    logger.info("="*60)
    
    return stats