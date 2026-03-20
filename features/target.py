import pandas as pd
import numpy as np
import logging
from pathlib import Path


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "target_engine.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def create_targets(df: pd.DataFrame, smoothing=True) -> pd.DataFrame:
    """
    Create forward-looking target variables for supervised learning.
    
    IMPROVED VERSION with:
    - Optional smoothing to reduce noise
    - Better handling of edge cases
    - Direction confidence metric
    
    Generates 5-day ahead returns and binary direction labels by looking
    forward in time within each ticker's time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns: Date, Ticker, Close.
        Should be sorted by Ticker and Date for optimal performance.
    smoothing : bool, default=True
        If True, uses 3-day MA of close prices to reduce noise in targets.
        This helps models learn cleaner patterns.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus target columns:
        - future_return_5d: return from current to 5 days ahead
        - direction_5d: binary indicator (1 if future_return_5d > 0, else 0)
        - direction_confidence: absolute value of future_return_5d (strength of move)
        
    Raises
    ------
    ValueError
        If required columns (Date, Ticker, Close) are missing.
        
    Notes
    -----
    - Smoothing helps reduce noise in target but maintains predictive power
    - NaN values exist for last 5 rows of each ticker (expected behavior)
    - Direction confidence helps weighted ensemble methods
    """
    logger.info("Starting IMPROVED target generation")
    logger.info(f"Smoothing enabled: {smoothing}")
    
    required_columns = ['Date', 'Ticker', 'Close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    df_out = df.copy()
    initial_rows = len(df_out)
    logger.info(f"Processing {initial_rows} rows")
    
    if not pd.api.types.is_datetime64_any_dtype(df_out['Date']):
        logger.warning("Date column is not datetime type, attempting conversion")
        df_out['Date'] = pd.to_datetime(df_out['Date'])
    
    df_out = df_out.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # Optional smoothing to reduce noise
    if smoothing:
        logger.info("Applying 3-day MA smoothing to close prices for target calculation")
        df_out['close_smooth'] = df_out.groupby('Ticker')['Close'].transform(
            lambda x: x.rolling(window=3, min_periods=1, center=False).mean()
        )
        price_column = 'close_smooth'
    else:
        logger.info("Using raw close prices for target calculation")
        price_column = 'Close'
    
    logger.info("Computing future_return_5d (5-day forward return)")
    
    # Get future price (5 days ahead)
    df_out['future_price_5d'] = df_out.groupby('Ticker')[price_column].shift(-5)
    
    # Calculate return
    df_out['future_return_5d'] = np.log(
        df_out['future_price_5d'] / df_out[price_column]
    )
    
    # Replace infinities with NaN (can happen if price is 0)
    df_out['future_return_5d'] = df_out['future_return_5d'].replace([np.inf, -np.inf], np.nan)
    
    # Clean up intermediate column
    df_out = df_out.drop(columns=['future_price_5d'])
    if smoothing:
        df_out = df_out.drop(columns=['close_smooth'])
    
    logger.info("Computing direction_5d (binary direction indicator)")
    df_out['direction_5d'] = np.where(
        df_out['future_return_5d'] > 0,
        1,
        np.where(
            df_out['future_return_5d'].isna(),
            np.nan,
            0
        )
    )
    
    logger.info("Computing direction_confidence (magnitude of move)")
    df_out['direction_confidence'] = np.abs(df_out['future_return_5d'])
    
    # Statistics
    nan_count_return = df_out['future_return_5d'].isna().sum()
    nan_count_direction = df_out['direction_5d'].isna().sum()
    nan_pct = 100 * nan_count_return / len(df_out)
    
    logger.info(f"future_return_5d: {nan_count_return} NaN values ({nan_pct:.2f}%)")
    logger.info(f"direction_5d: {nan_count_direction} NaN values ({nan_pct:.2f}%)")
    
    # Validate expected NaN count
    tickers_count = df_out['Ticker'].nunique()
    expected_nan_per_ticker = 5
    expected_total_nan = tickers_count * expected_nan_per_ticker
    
    logger.info(f"Expected NaN count: ~{expected_total_nan} ({tickers_count} tickers × 5 days)")
    logger.info(f"Actual NaN count: {nan_count_return}")
    
    # Target distribution statistics
    valid_returns = df_out['future_return_5d'].dropna()
    if len(valid_returns) > 0:
        logger.info(f"Target statistics:")
        logger.info(f"  Mean: {valid_returns.mean():.6f}")
        logger.info(f"  Std: {valid_returns.std():.6f}")
        logger.info(f"  Min: {valid_returns.min():.6f}")
        logger.info(f"  Max: {valid_returns.max():.6f}")
        
        positive_count = (valid_returns > 0).sum()
        positive_pct = 100 * positive_count / len(valid_returns)
        logger.info(f"  Positive returns: {positive_count} ({positive_pct:.2f}%)")
    
    df_out = df_out.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    logger.info(f"IMPROVED target generation complete. Output shape: {df_out.shape}")
    
    return df_out
