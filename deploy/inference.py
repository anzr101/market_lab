import pandas as pd
import logging
from pathlib import Path
from typing import Dict
import numpy as np

from models.save_load import load_models
from models.predict import predict



log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "inference.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def run_inference(model_dir: str, X: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Run inference using trained models on new data.
    
    Loads all trained models from specified directory and generates predictions
    for the provided feature matrix. Returns predictions from all models as
    a dictionary mapping model names to prediction arrays.
    
    This is the primary entry point for making predictions with saved models.
    Combines model loading and prediction generation in a single pipeline.
    
    Parameters
    ----------
    model_dir : str
        Directory path containing saved model files (.pkl).
        Must exist and contain at least one valid model file.
    X : pd.DataFrame
        Feature matrix for prediction.
        Should contain numeric features matching the training data schema.
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping model names to prediction arrays.
        Each array has shape (n_samples,).
        Keys correspond to model filenames without .pkl extension.
        
    Raises
    ------
    ValueError
        If model_dir is invalid, X is empty, or no models loaded.
    TypeError
        If model_dir is not a string or X is not a DataFrame.
    FileNotFoundError
        If model_dir does not exist.
    Exception
        If model loading or prediction generation fails.
        
    Notes
    -----
    - Uses load_models() from save_load module for model loading
    - Uses predict() from predict module for prediction generation
    - All models must be compatible with the feature matrix X
    - NaN values in X are automatically handled by predict()
    - Logs detailed information about loaded models and predictions
    
    Examples
    --------
    >>> X_new = pd.DataFrame({
    ...     'feature1': [1.0, 2.0, 3.0],
    ...     'feature2': [4.0, 5.0, 6.0]
    ... })
    >>> predictions = run_inference('saved_models', X_new)
    >>> print(predictions.keys())
    dict_keys(['RandomForestRegressor', 'LGBMRegressor'])
    """
    logger.info("="*60)
    logger.info("STARTING INFERENCE PIPELINE")
    logger.info("="*60)
    
    if not isinstance(model_dir, str):
        logger.error("model_dir must be a string")
        raise TypeError("model_dir must be a string path")
    
    if not model_dir:
        logger.error("Empty model_dir provided")
        raise ValueError("model_dir cannot be empty")
    
    if not isinstance(X, pd.DataFrame):
        logger.error("X must be a pandas DataFrame")
        raise TypeError("X must be a pandas DataFrame")
    
    if len(X) == 0:
        logger.error("Empty DataFrame provided")
        raise ValueError("Feature DataFrame X cannot be empty")
    
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Input feature matrix shape: {X.shape}")
    
    model_dir_path = Path(model_dir)
    
    if not model_dir_path.exists():
        logger.error(f"Model directory does not exist: {model_dir_path.absolute()}")
        raise FileNotFoundError(f"Model directory not found: {model_dir_path.absolute()}")
    
    logger.info("Step 1: Loading models from directory")
    
    try:
        models = load_models(model_dir)
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise
    
    if len(models) == 0:
        logger.error("No models loaded from directory")
        raise ValueError(f"No models found in directory: {model_dir}")
    
    logger.info(f"Successfully loaded {len(models)} models: {list(models.keys())}")
    
    logger.info("Step 2: Generating predictions")
    
    try:
        predictions = predict(models, X)
    except Exception as e:
        logger.error(f"Failed to generate predictions: {str(e)}")
        raise
    
    if len(predictions) == 0:
        logger.error("No predictions generated")
        raise ValueError("Prediction generation returned empty results")
    
    logger.info(f"Successfully generated predictions for {len(predictions)} models")
    
    for model_name, pred_array in predictions.items():
        pred_stats = {
            'count': len(pred_array),
            'mean': pred_array.mean(),
            'std': pred_array.std(),
            'min': pred_array.min(),
            'max': pred_array.max()
        }
        logger.info(f"Model '{model_name}' predictions: {pred_stats}")
    
    logger.info("="*60)
    logger.info("INFERENCE PIPELINE COMPLETE")
    logger.info("="*60)
    
    return predictions