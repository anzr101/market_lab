import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "backtest.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def run_backtest(df: pd.DataFrame, predictions: np.ndarray) -> Dict[str, Any]:
    """
    Simulate portfolio backtest using predicted returns.
    
    Constructs a long-short portfolio based on predicted returns, normalizes
    positions daily, and evaluates performance through cumulative returns,
    Sharpe ratio, and maximum drawdown metrics.
    
    Portfolio construction:
    1. Positions set equal to predictions
    2. Daily normalization: each position divided by sum of absolute positions
    3. Actual returns computed as next-day log returns per ticker
    4. Portfolio return per day = sum(normalized_position * actual_return)
    5. Equity curve built from cumulative product of (1 + daily_return)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least Date, Ticker, Close columns.
        Should be sorted by Date and Ticker.
    predictions : np.ndarray
        Array of predicted returns aligned with df rows.
        Length must match len(df).
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - equity_curve: np.ndarray of cumulative equity values
        - cumulative_return: float, total return over period
        - sharpe: float, annualized Sharpe ratio
        - max_drawdown: float, maximum drawdown percentage
        
    Raises
    ------
    ValueError
        If required columns missing or predictions length mismatch.
        
    Notes
    -----
    - Positions normalized daily by sum of absolute values
    - Portfolio returns computed as weighted sum of actual returns
    - Actual returns are next-day log returns per ticker
    - Last day of each ticker excluded (no future return available)
    - Sharpe ratio annualized assuming 252 trading days
    - Max drawdown computed as largest peak-to-trough decline
    """
    logger.info("Starting backtest simulation")
    
    required_columns = ['Date', 'Ticker', 'Close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if len(predictions) != len(df):
        logger.error(f"Predictions length {len(predictions)} != DataFrame length {len(df)}")
        raise ValueError(f"Predictions length must match DataFrame length")
    
    df_bt = df.copy()
    df_bt['prediction'] = predictions
    
    logger.info(f"Backtest data: {len(df_bt)} rows, {df_bt['Ticker'].nunique()} tickers")
    
    if not pd.api.types.is_datetime64_any_dtype(df_bt['Date']):
        df_bt['Date'] = pd.to_datetime(df_bt['Date'])
    
    df_bt = df_bt.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    logger.info("Computing actual next-day returns per ticker")
    df_bt['next_close'] = df_bt.groupby('Ticker')['Close'].shift(-1)
    df_bt['actual_return'] = np.log(df_bt['next_close'] / df_bt['Close'])
    
    rows_before = len(df_bt)
    df_bt = df_bt.dropna(subset=['actual_return'])
    rows_after = len(df_bt)
    logger.info(f"Removed {rows_before - rows_after} rows without future returns ({rows_after} remaining)")
    
    if len(df_bt) == 0:
        logger.error("No valid returns available for backtest")
        raise ValueError("Insufficient data: no valid next-day returns")
    
    df_bt['position'] = df_bt['prediction']
    
    logger.info("Normalizing positions daily")
    position_abs_sum = df_bt.groupby('Date')['position'].transform(lambda x: x.abs().sum())
    df_bt['position_normalized'] = np.where(
        position_abs_sum != 0,
        df_bt['position'] / position_abs_sum,
        0
    )
    
    zero_position_days = (position_abs_sum == 0).sum()
    if zero_position_days > 0:
        logger.warning(f"{zero_position_days} dates have zero total positions")
    
    logger.info("Computing weighted returns")
    df_bt['weighted_return'] = df_bt['position_normalized'] * df_bt['actual_return']
    
    daily_returns = df_bt.groupby('Date')['weighted_return'].sum().sort_index()
    
    logger.info(f"Daily portfolio returns computed: {len(daily_returns)} trading days")
    
    if len(daily_returns) == 0:
        logger.error("No daily returns computed")
        raise ValueError("Failed to compute daily returns")
    
    logger.info("Building equity curve")
    equity_curve = (1 + daily_returns).cumprod().values
    
    cumulative_return = equity_curve[-1] - 1
    logger.info(f"Cumulative return: {cumulative_return:.6f} ({cumulative_return*100:.2f}%)")
    
    logger.info("Computing Sharpe ratio")
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    
    if std_return != 0 and not np.isnan(std_return):
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
        logger.warning("Daily returns std is zero or NaN, Sharpe ratio set to 0")
    
    logger.info(f"Sharpe ratio (annualized): {sharpe_ratio:.4f}")
    
    logger.info("Computing maximum drawdown")
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_drawdown = drawdowns.min()
    
    logger.info(f"Maximum drawdown: {max_drawdown:.6f} ({max_drawdown*100:.2f}%)")
    
    results = {
        'equity_curve': equity_curve,
        'cumulative_return': float(cumulative_return),
        'sharpe': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown)
    }
    
    logger.info("="*60)
    logger.info("BACKTEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Trading Days: {len(daily_returns)}")
    logger.info(f"Cumulative Return: {cumulative_return*100:.2f}%")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {max_drawdown*100:.2f}%")
    logger.info("="*60)
    
    logger.info("Backtest simulation complete")
    
    return results