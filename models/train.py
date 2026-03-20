import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "model_train.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _create_sequences(X, y, window_size=10):
    """
    Create sequences for LSTM training.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    window_size : int
        Size of sequence window.
        
    Returns
    -------
    X_seq : np.ndarray
        Reshaped sequences (samples, window, features).
    y_seq : np.ndarray
        Corresponding targets.
    """
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size - 1])
    
    return np.array(X_seq), np.array(y_seq)


def train_models(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Train multiple regression models on time-series features.
    
    Performs chronological train/test split, trains three regression models,
    and evaluates performance metrics. Maintains strict time-series safety
    by ensuring test data comes after train data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features and target column 'future_return_5d'.
        Should contain numeric features and may contain NaN values in target.
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping model names to their results:
        {
            "model_name": {
                "model": trained_model_object,
                "mae": mean_absolute_error_on_test,
                "r2": r2_score_on_test
            }
        }
        
    Raises
    ------
    ValueError
        If target column 'future_return_5d' is missing.
        If insufficient non-NaN data available for training.
        
    Notes
    -----
    - Removes rows where target is NaN before splitting
    - Uses chronological 80/20 train/test split (no shuffling)
    - Automatically identifies numeric feature columns
    - Excludes Date, Ticker, future_return_5d, direction_5d from features
    - Models are trained with deterministic random states for reproducibility
    """
    logger.info("Starting model training pipeline")
    
    target_column = 'future_return_5d'
    
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in DataFrame")
        raise ValueError(f"Target column '{target_column}' is required but missing")
    
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    logger.info(f"Initial dataset: {initial_rows} rows")
    
    df_clean = df_clean.dropna(subset=[target_column])
    rows_after_dropna = len(df_clean)
    rows_dropped = initial_rows - rows_after_dropna
    
    logger.info(f"Removed {rows_dropped} rows with NaN target ({rows_after_dropna} remaining)")
    
    if rows_after_dropna == 0:
        logger.error("No valid rows remaining after removing NaN targets")
        raise ValueError("Insufficient data: all target values are NaN")
    
    if 'Date' in df_clean.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_clean['Date']):
            df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        df_clean = df_clean.sort_values('Date').reset_index(drop=True)
        logger.info("Data sorted chronologically by Date")
    
    exclude_columns = {'Date', 'Ticker', 'future_return_5d', 'direction_5d'}
    
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    feature_columns = [col for col in numeric_columns if col not in exclude_columns]
    
    if len(feature_columns) == 0:
        logger.error("No valid feature columns found")
        raise ValueError("No numeric feature columns available for training")
    
    logger.info(f"Identified {len(feature_columns)} feature columns: {feature_columns}")
    
    X = df_clean[feature_columns].copy()
    y = df_clean[target_column].copy()
    
    nan_features = X.isna().sum().sum()
    if nan_features > 0:
        logger.warning(f"Found {nan_features} NaN values in features, filling with 0")
        X = X.fillna(0)
    
    split_index = int(0.8 * len(df_clean))
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    logger.info(f"Chronological split: train={len(X_train)} rows, test={len(X_test)} rows")
    logger.info(f"Split ratio: {100*len(X_train)/len(df_clean):.1f}% train, {100*len(X_test)/len(df_clean):.1f}% test")
    
    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("Train or test set is empty after split")
        raise ValueError("Insufficient data for train/test split")
    
    results = {}
    
    logger.info("Training LinearRegression")
    try:
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        
        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        r2_lr = r2_score(y_test, y_pred_lr)
        
        results['LinearRegression'] = {
            'model': lr_model,
            'mae': mae_lr,
            'r2': r2_lr
        }
        logger.info(f"LinearRegression - MAE: {mae_lr:.6f}, R2: {r2_lr:.6f}")
    except Exception as e:
        logger.error(f"LinearRegression training failed: {str(e)}")
        raise
    
    logger.info("Training RandomForestRegressor")
    try:
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)
        
        results['RandomForestRegressor'] = {
            'model': rf_model,
            'mae': mae_rf,
            'r2': r2_rf
        }
        logger.info(f"RandomForestRegressor - MAE: {mae_rf:.6f}, R2: {r2_rf:.6f}")
    except Exception as e:
        logger.error(f"RandomForestRegressor training failed: {str(e)}")
        raise
    
    logger.info("Training LGBMRegressor")
    try:
        lgbm_model = LGBMRegressor(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgbm_model.fit(X_train, y_train)
        y_pred_lgbm = lgbm_model.predict(X_test)
        
        mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
        r2_lgbm = r2_score(y_test, y_pred_lgbm)
        
        results['LGBMRegressor'] = {
            'model': lgbm_model,
            'mae': mae_lgbm,
            'r2': r2_lgbm
        }
        logger.info(f"LGBMRegressor - MAE: {mae_lgbm:.6f}, R2: {r2_lgbm:.6f}")
    except Exception as e:
        logger.error(f"LGBMRegressor training failed: {str(e)}")
        raise
    
    logger.info("Training GradientBoostingRegressor")
    try:
        gb_model = GradientBoostingRegressor(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        y_pred_gb = gb_model.predict(X_test)
        
        mae_gb = mean_absolute_error(y_test, y_pred_gb)
        r2_gb = r2_score(y_test, y_pred_gb)
        
        results['GradientBoostingRegressor'] = {
            'model': gb_model,
            'mae': mae_gb,
            'r2': r2_gb
        }
        logger.info(f"GradientBoostingRegressor - MAE: {mae_gb:.6f}, R2: {r2_gb:.6f}")
    except Exception as e:
        logger.error(f"GradientBoostingRegressor training failed: {str(e)}")
        raise
    
    logger.info("Training SVR")
    try:
        X_train_svr = X_train
        y_train_svr = y_train
        
        if len(X_train) > 15000:
            logger.info(f"Dataset large ({len(X_train)} rows), subsampling to 12000 rows for SVR")
            np.random.seed(42)
            sample_indices = np.random.choice(len(X_train), size=12000, replace=False)
            X_train_svr = X_train.iloc[sample_indices]
            y_train_svr = y_train.iloc[sample_indices]
            logger.info(f"SVR training on {len(X_train_svr)} subsampled rows")
        
        svr_model = SVR(
            kernel='rbf',
            C=1.0,
            epsilon=0.1
        )
        svr_model.fit(X_train_svr, y_train_svr)
        y_pred_svr = svr_model.predict(X_test)
        
        mae_svr = mean_absolute_error(y_test, y_pred_svr)
        r2_svr = r2_score(y_test, y_pred_svr)
        
        results['SVR'] = {
            'model': svr_model,
            'mae': mae_svr,
            'r2': r2_svr
        }
        logger.info(f"SVR - MAE: {mae_svr:.6f}, R2: {r2_svr:.6f}")
    except Exception as e:
        logger.error(f"SVR training failed: {str(e)}")
        raise
    
    logger.info("Training LSTM")
    try:
        window_size = 10
        
        if len(X_train) < window_size:
            logger.warning(f"Dataset too small ({len(X_train)} rows) for LSTM window size {window_size}, skipping LSTM")
        else:
            tf.random.set_seed(42)
            
            X_train_array = X_train.values
            y_train_array = y_train.values
            X_test_array = X_test.values
            y_test_array = y_test.values
            
            X_train_seq, y_train_seq = _create_sequences(X_train_array, y_train_array, window_size)
            X_test_seq, y_test_seq = _create_sequences(X_test_array, y_test_array, window_size)
            
            logger.info(f"LSTM sequences created: train={X_train_seq.shape}, test={X_test_seq.shape}")
            
            lstm_model = Sequential([
                LSTM(32, input_shape=(window_size, X_train.shape[1])),
                Dense(1)
            ])
            
            lstm_model.compile(optimizer='adam', loss='mae')
            
            lstm_model.fit(
                X_train_seq,
                y_train_seq,
                epochs=5,
                batch_size=128,
                verbose=0
            )
            
            y_pred_lstm_seq = lstm_model.predict(X_test_seq, verbose=0)
            y_pred_lstm = y_pred_lstm_seq.flatten()
            
            mae_lstm = mean_absolute_error(y_test_seq, y_pred_lstm)
            r2_lstm = r2_score(y_test_seq, y_pred_lstm)
            
            results['LSTM'] = {
                'model': lstm_model,
                'mae': mae_lstm,
                'r2': r2_lstm
            }
            logger.info(f"LSTM - MAE: {mae_lstm:.6f}, R2: {r2_lstm:.6f}")
    except Exception as e:
        logger.error(f"LSTM training failed: {str(e)}")
        raise
    
    best_model = min(results.items(), key=lambda x: x[1]['mae'])
    logger.info(f"Best model by MAE: {best_model[0]} (MAE: {best_model[1]['mae']:.6f})")
    
    logger.info("Model training pipeline complete")
    
    return results