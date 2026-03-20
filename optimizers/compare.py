import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any

from data.loader import load_market_data
from features.generator import generate_features
from features.target import create_targets
from optimizers.optimizer import run_optimizers


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "optimizer_compare.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def run_optimizer_comparison(data_path: str) -> Dict[str, Any]:

    logger.info("="*60)
    logger.info("STARTING OPTIMIZER COMPARISON PIPELINE")
    logger.info("="*60)
    
    if not data_path or not isinstance(data_path, str):
        logger.error("Invalid data_path provided")
        raise ValueError("data_path must be a non-empty string")
    
    logger.info(f"Data path: {data_path}")
    
    logger.info("Step 1: Loading market data")
    df = load_market_data(data_path)
    logger.info(f"Data loaded: {df.shape}")
    
    logger.info("Step 2: Generating features")
    df = generate_features(df)
    logger.info(f"Features generated: {df.shape}")
    
    logger.info("Step 3: Creating targets")
    df = create_targets(df)
    logger.info(f"Targets created: {df.shape}")
    
    logger.info("Step 4: Dropping NaN targets")
    initial_rows = len(df)
    df = df.dropna(subset=['future_return_5d'])
    rows_dropped = initial_rows - len(df)
    logger.info(f"Dropped {rows_dropped} rows with NaN targets ({len(df)} remaining)")
    
    if len(df) == 0:
        logger.error("No valid data remaining after dropping NaN targets")
        raise ValueError("Insufficient data: all targets are NaN")
    
    logger.info("Step 5: Sorting chronologically")
    if 'Date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        logger.info("Data sorted by Date")
    
    logger.info("Step 6: Selecting numeric features")
    exclude_columns = {'Date', 'Ticker', 'future_return_5d', 'direction_5d'}
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in numeric_columns if col not in exclude_columns]
    
    if len(feature_columns) == 0:
        logger.error("No valid feature columns found")
        raise ValueError("No numeric feature columns available")
    
    logger.info(f"Selected {len(feature_columns)} features")
    
    X = df[feature_columns].fillna(0)
    y = df['future_return_5d']
    
    logger.info("Step 7: Splitting data chronologically (80/20)")
    split_index = int(0.8 * len(df))
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    train_size = len(X_train)
    test_size = len(X_test)
    
    logger.info(f"Train size: {train_size}")
    logger.info(f"Test size: {test_size}")
    
    if train_size == 0 or test_size == 0:
        logger.error("Train or test set is empty")
        raise ValueError("Insufficient data for train/test split")
    
    logger.info("Step 8: Running optimizer comparison")
    optimizer_results = run_optimizers(X_train, y_train, X_test, y_test)
    
    logger.info("="*60)
    logger.info("OPTIMIZER COMPARISON RESULTS")
    logger.info("="*60)
    
    for optimizer_name, result in optimizer_results.items():
        logger.info(f"{optimizer_name}: MAE = {result['mae']:.6f}")
    
    best_optimizer = min(optimizer_results.items(), key=lambda x: x[1]['mae'])

    logger.info(f"Best Optimizer: {best_optimizer[0]} (MAE: {best_optimizer[1]['mae']:.6f})")
    logger.info("="*60)
    
    results = {
        'optimizer_results': optimizer_results,
        'train_size': train_size,
        'test_size': test_size,
        'feature_count': len(feature_columns)
    }
    
    logger.info("Optimizer comparison pipeline complete")
    
    return results
