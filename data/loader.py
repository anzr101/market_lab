import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import List


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "data_loader.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_market_data(file_path: str) -> pd.DataFrame:
    """
    Load and validate raw market OHLCV data from CSV.
    
    This function performs comprehensive data validation and cleaning to ensure
    time-series safety and data integrity for downstream modeling.
    
    Parameters
    ----------
    file_path : str
        Path to raw CSV file containing market data.
        
    Returns
    -------
    pd.DataFrame
        Cleaned and validated DataFrame with columns:
        [Date, Ticker, Open, High, Low, Close, Volume]
        Sorted by Date then Ticker.
        
    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If required columns are missing or data validation fails.
    """
    logger.info(f"Starting data load from: {file_path}")
    
    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path, low_memory=False)

        logger.info(f"Loaded CSV with {len(df)} rows")
    except Exception as e:
        logger.error(f"Failed to load CSV: {str(e)}")
        raise
    
    initial_rows = len(df)
    
    df.columns = df.columns.str.strip().str.title()
    
    required_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    df = df[required_columns].copy()
    
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    except Exception as e:
        logger.error(f"Failed to convert Date column: {str(e)}")
        raise ValueError(f"Date column conversion failed: {str(e)}")
    
    null_dates = df['Date'].isna().sum()
    if null_dates > 0:
        logger.warning(f"Removing {null_dates} rows with null dates")
        df = df.dropna(subset=['Date'])
    
    today = pd.Timestamp.now().normalize()
    future_dates = (df['Date'] > today).sum()
    if future_dates > 0:
        logger.warning(f"Removing {future_dates} rows with future dates")
        df = df[df['Date'] <= today]
    
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    inf_mask = np.isinf(df[numeric_columns]).any(axis=1)
    inf_count = inf_mask.sum()
    if inf_count > 0:
        logger.warning(f"Removing {inf_count} rows with infinite values")
        df = df[~inf_mask]
    
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    duplicates = df.duplicated(subset=['Date', 'Ticker'], keep='first').sum()
    if duplicates > 0:
        logger.warning(f"Removing {duplicates} duplicate rows")
        df = df.drop_duplicates(subset=['Date', 'Ticker'], keep='first')
    
    price_valid = (
        (df['Low'] <= df['Open']) & 
        (df['Open'] <= df['High']) & 
        (df['Low'] <= df['Close']) & 
        (df['Close'] <= df['High'])
    )
    invalid_prices = (~price_valid).sum()
    if invalid_prices > 0:
        logger.warning(f"Removing {invalid_prices} rows with invalid price logic")
        df = df[price_valid]
    
    invalid_volume = (df['Volume'] < 0).sum()
    if invalid_volume > 0:
        logger.warning(f"Removing {invalid_volume} rows with negative volume")
        df = df[df['Volume'] >= 0]
    
    price_columns = ['Open', 'High', 'Low', 'Close']
    for col in price_columns:
        df[col] = df.groupby('Ticker')[col].ffill()
    
    df['Volume'] = df['Volume'].fillna(0)
    
    rows_before_dropna = len(df)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    rows_dropped_na = rows_before_dropna - len(df)
    if rows_dropped_na > 0:
        logger.warning(f"Dropped {rows_dropped_na} rows with remaining NaN values")
    
    df = df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    final_rows = len(df)
    rows_removed = initial_rows - final_rows
    
    logger.info(f"Data cleaning complete: {initial_rows} -> {final_rows} rows ({rows_removed} removed)")
    
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "processed_data.parquet"
    
    try:
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved cleaned data to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save parquet: {str(e)}")
        raise
    
    return df