import pandas as pd
import numpy as np
import logging
from pathlib import Path


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "feature_generator.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate technical features for time-series modeling.
    
    Creates rolling statistics, momentum indicators, and cross-sectional
    features while maintaining strict time-series safety (no lookahead bias).
    All features use only historical data and are safe for walk-forward modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns: Date, Ticker, Open, High, Low, Close, Volume.
        Should be sorted by Ticker and Date for optimal performance.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus 10 new feature columns:
        - log_return: logarithmic return
        - rolling_mean_5: 5-period rolling mean of Close
        - rolling_std_5: 5-period rolling standard deviation of Close
        - ma_distance: distance from 5-period moving average
        - ma_slope: 3-period change in moving average
        - volatility_ratio: current volatility relative to 10-period average
        - daily_rank: cross-sectional rank of returns per date
        - return_zscore: cross-sectional z-score of returns per date
        - volume_zscore: 20-period rolling z-score of volume per ticker
        - price_range: normalized daily price range
        
    Raises
    ------
    ValueError
        If required columns are missing.
        
    Notes
    -----
    Features with rolling windows will have NaN values for initial periods
    where insufficient historical data exists. This is expected and safe.
    """
    logger.info("Starting feature generation")
    
    required_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
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
    
    logger.info("Computing log_return")
    df_out['log_return'] = df_out.groupby('Ticker')['Close'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    
    logger.info("Computing rolling_mean_5")
    df_out['rolling_mean_5'] = df_out.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(window=5, min_periods=5).mean()
    )
    
    logger.info("Computing rolling_std_5")
    df_out['rolling_std_5'] = df_out.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(window=5, min_periods=5).std()
    )
    
    logger.info("Computing ma_distance")
    df_out['ma_distance'] = df_out['Close'] - df_out['rolling_mean_5']
    
    logger.info("Computing ma_slope")
    df_out['ma_slope'] = df_out.groupby('Ticker')['rolling_mean_5'].transform(
        lambda x: x - x.shift(3)
    )
    
    logger.info("Computing volatility_ratio")
    rolling_std_mean_10 = df_out.groupby('Ticker')['rolling_std_5'].transform(
        lambda x: x.rolling(window=10, min_periods=10).mean()
    )
    df_out['volatility_ratio'] = np.where(
        rolling_std_mean_10 != 0,
        df_out['rolling_std_5'] / rolling_std_mean_10,
        np.nan
    )
    
    logger.info("Computing daily_rank")
    df_out['daily_rank'] = df_out.groupby('Date')['log_return'].rank(
        method='average', na_option='keep'
    )
    
    logger.info("Computing return_zscore")
    df_out['return_zscore'] = df_out.groupby('Date')['log_return'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    logger.info("Computing volume_zscore")
    volume_rolling_mean = df_out.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(window=20, min_periods=20).mean()
    )
    volume_rolling_std = df_out.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(window=20, min_periods=20).std()
    )
    df_out['volume_zscore'] = np.where(
        volume_rolling_std != 0,
        (df_out['Volume'] - volume_rolling_mean) / volume_rolling_std,
        np.nan
    )
    
    logger.info("Computing price_range")
    df_out['price_range'] = np.where(
        df_out['Close'] != 0,
        (df_out['High'] - df_out['Low']) / df_out['Close'],
        np.nan
    )
    
    df_out = df_out.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    feature_columns = [
        'log_return', 'rolling_mean_5', 'rolling_std_5', 'ma_distance',
        'ma_slope', 'volatility_ratio', 'daily_rank', 'return_zscore',
        'volume_zscore', 'price_range'
    ]
    
    for col in feature_columns:
        nan_count = df_out[col].isna().sum()
        nan_pct = 100 * nan_count / len(df_out)
        logger.info(f"Feature '{col}': {nan_count} NaN values ({nan_pct:.2f}%)")
    
    logger.info(f"Feature generation complete. Output shape: {df_out.shape}")
    
    return df_out