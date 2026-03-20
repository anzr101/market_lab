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
    handler = logging.FileHandler(log_dir / "predict.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


WINDOW_SIZE = 10


def _create_sequences(X, window_size=10):
    """
    Create rolling sequences for LSTM prediction.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    window_size : int
        Size of sequence window.
        
    Returns
    -------
    np.ndarray
        Sequences of shape (n_samples - window_size + 1, window_size, n_features).
    """
    seq = []
    for i in range(len(X) - window_size + 1):
        seq.append(X[i:i + window_size])
    return np.array(seq)


def predict(models: Dict[str, Any], X: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Generate predictions from multiple trained models.
    
    Applies each model in the dictionary to the input features and returns
    predictions as numpy arrays. Handles missing values by filling with zero
    and ensures all features are numeric before prediction.
    
    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary mapping model names to trained model objects.
        Each model must have a predict() method.
    X : pd.DataFrame
        Feature matrix for prediction.
        Should contain only numeric columns.
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping model names to prediction arrays.
        Each array has shape (n_samples,).
        
    Raises
    ------
    ValueError
        If models dict is empty, X is empty, or X contains no numeric columns.
    TypeError
        If models is not a dict or X is not a DataFrame.
    AttributeError
        If any model does not have a predict() method.
    Exception
        If prediction fails for any model.
        
    Notes
    -----
    - NaN values in X are filled with 0 before prediction
    - Only numeric columns are used for prediction
    - Non-numeric columns are automatically dropped with warning
    - All models must accept the same feature matrix shape
    - LSTM models require sequence input and output is padded with NaNs
    """
    logger.info("Starting prediction generation")
    
    if not isinstance(models, dict):
        logger.error("Models must be a dictionary")
        raise TypeError("Models must be a dictionary mapping names to model objects")
    
    if len(models) == 0:
        logger.error("Empty models dictionary provided")
        raise ValueError("Models dictionary cannot be empty")
    
    if not isinstance(X, pd.DataFrame):
        logger.error("X must be a pandas DataFrame")
        raise TypeError("X must be a pandas DataFrame")
    
    if len(X) == 0:
        logger.error("Empty DataFrame provided")
        raise ValueError("Feature DataFrame X cannot be empty")
    
    logger.info(f"Input shape: {X.shape}")
    logger.info(f"Number of models: {len(models)}")
    
    X_pred = X.copy()
    
    numeric_columns = X_pred.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) == 0:
        logger.error("No numeric columns found in X")
        raise ValueError("X must contain at least one numeric column")
    
    non_numeric_columns = [col for col in X_pred.columns if col not in numeric_columns]
    
    if non_numeric_columns:
        logger.warning(f"Dropping {len(non_numeric_columns)} non-numeric columns: {non_numeric_columns}")
        X_pred = X_pred[numeric_columns]
    
    logger.info(f"Using {len(numeric_columns)} numeric features for prediction")
    
    nan_count = X_pred.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in features, filling with 0")
        X_pred = X_pred.fillna(0)
    
    inf_count = np.isinf(X_pred.values).sum()
    if inf_count > 0:
        logger.warning(f"Found {inf_count} infinite values in features, replacing with 0")
        X_pred = X_pred.replace([np.inf, -np.inf], 0)
    
    predictions = {}
    success_count = 0
    
    for model_name, model_object in models.items():
        logger.info(f"Generating predictions for model: {model_name}")
        
        if not hasattr(model_object, 'predict'):
            logger.error(f"Model '{model_name}' does not have predict() method")
            raise AttributeError(f"Model '{model_name}' must have a predict() method")
        
        try:
            if model_name == "LSTM":
                logger.info(f"LSTM model detected, creating sequences with window_size={WINDOW_SIZE}")
                
                if len(X_pred) < WINDOW_SIZE:
                    logger.error(f"Insufficient data for LSTM sequences: {len(X_pred)} < {WINDOW_SIZE}")
                    raise ValueError(f"LSTM requires at least {WINDOW_SIZE} samples, got {len(X_pred)}")
                
                X_array = X_pred.values
                X_seq = _create_sequences(X_array, WINDOW_SIZE)
                
                logger.info(f"Created sequences: shape={X_seq.shape}")
                
                y_pred_seq = model_object.predict(X_seq, verbose=0)
                y_pred_seq = y_pred_seq.flatten()
                
                nan_padding = np.full(WINDOW_SIZE - 1, np.nan)
                y_pred = np.concatenate([nan_padding, y_pred_seq])
                
                logger.info(f"LSTM predictions: {len(y_pred_seq)} sequences + {len(nan_padding)} NaN padding = {len(y_pred)} total")
            else:
                y_pred = model_object.predict(X_pred)
            
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred)
            
            if len(y_pred) != len(X_pred):
                logger.error(f"Prediction length mismatch for '{model_name}': {len(y_pred)} != {len(X_pred)}")
                raise ValueError(f"Prediction length mismatch for model '{model_name}'")
            
            predictions[model_name] = y_pred
            
            valid_predictions = y_pred[~np.isnan(y_pred)]
            if len(valid_predictions) > 0:
                logger.info(f"Model '{model_name}': generated {len(y_pred)} predictions ({len(valid_predictions)} valid)")
                logger.info(f"Model '{model_name}': prediction stats - mean={valid_predictions.mean():.6f}, std={valid_predictions.std():.6f}")
            else:
                logger.info(f"Model '{model_name}': generated {len(y_pred)} predictions (all NaN)")
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Prediction failed for model '{model_name}': {str(e)}")
            raise Exception(f"Failed to generate predictions for model '{model_name}': {str(e)}")
    
    logger.info(f"Successfully generated predictions for {success_count}/{len(models)} models")
    logger.info("Prediction generation complete")
    
    return predictions