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


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create forward-looking target variables for supervised learning.
    
    Generates 5-day ahead returns and binary direction labels by looking
    forward in time within each ticker's time series. This introduces NaN
    values at the end of each ticker's series where future data is unavailable,
    which is expected and safe for train/test splitting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns: Date, Ticker, Close.
        Should be sorted by Ticker and Date for optimal performance.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus 2 target columns:
        - future_return_5d: log return from current close to close 5 days ahead
        - direction_5d: binary indicator (1 if future_return_5d > 0, else 0)
        
        NaN values will exist for the last 5 rows of each ticker where
        future data is unavailable. These rows should be excluded during
        model training.
        
    Raises
    ------
    ValueError
        If required columns (Date, Ticker, Close) are missing.
        
    Notes
    -----
    Target variables use future information and must never be used as features.
    Always split data chronologically and ensure targets are only computed
    on training data or with proper walk-forward logic.
    """
    logger.info("Starting target generation")
    
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
    
    logger.info("Computing future_return_5d (5-day forward return)")
    df_out['future_close_5d'] = df_out.groupby('Ticker')['Close'].shift(-5)
    
    df_out['future_return_5d'] = np.log(
        df_out['future_close_5d'] / df_out['Close']
    )
    
    df_out = df_out.drop(columns=['future_close_5d'])
    
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
    
    nan_count_return = df_out['future_return_5d'].isna().sum()
    nan_count_direction = df_out['direction_5d'].isna().sum()
    nan_pct = 100 * nan_count_return / len(df_out)
    
    logger.info(f"future_return_5d: {nan_count_return} NaN values ({nan_pct:.2f}%)")
    logger.info(f"direction_5d: {nan_count_direction} NaN values ({nan_pct:.2f}%)")
    
    tickers_count = df_out['Ticker'].nunique()
    expected_nan_per_ticker = 5
    expected_total_nan = tickers_count * expected_nan_per_ticker
    
    logger.info(f"Expected NaN count: ~{expected_total_nan} ({tickers_count} tickers × 5 days)")
    logger.info(f"Actual NaN count: {nan_count_return}")
    
    df_out = df_out.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    logger.info(f"Target generation complete. Output shape: {df_out.shape}")
    
    return df_out