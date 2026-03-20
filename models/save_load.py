import joblib
import logging
from pathlib import Path
from typing import Dict, Any


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "model_io.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def save_models(models: Dict[str, Any], folder: str) -> None:
    """
    Save trained models to disk using joblib serialization.
    
    Creates the target folder if it does not exist and saves each model
    as a separate pickle file. Model names become filenames with .pkl extension.
    Warns if overwriting existing files.
    
    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary mapping model names to model objects.
        Keys become filenames (with .pkl extension).
        Values must be serializable objects (scikit-learn models, etc.).
    folder : str
        Target directory path where models will be saved.
        Created if it does not exist.
        
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If models dictionary is empty or folder path is invalid.
    TypeError
        If models is not a dictionary.
    Exception
        If serialization fails for any model.
        
    Notes
    -----
    - Uses joblib for efficient serialization of large numpy arrays
    - Overwrites existing files with warning
    - All model objects must be picklable
    - Filenames sanitized to remove special characters
    """
    logger.info("Starting model save operation")
    
    if not isinstance(models, dict):
        logger.error("Models must be a dictionary")
        raise TypeError("Models must be a dictionary mapping names to model objects")
    
    if len(models) == 0:
        logger.error("Empty models dictionary provided")
        raise ValueError("Models dictionary cannot be empty")
    
    if not folder or not isinstance(folder, str):
        logger.error("Invalid folder path provided")
        raise ValueError("Folder path must be a non-empty string")
    
    folder_path = Path(folder)
    
    try:
        folder_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Target folder created/verified: {folder_path.absolute()}")
    except Exception as e:
        logger.error(f"Failed to create folder {folder}: {str(e)}")
        raise
    
    logger.info(f"Saving {len(models)} models to {folder_path.absolute()}")
    
    saved_count = 0
    
    for model_name, model_object in models.items():
        if not isinstance(model_name, str) or not model_name:
            logger.warning(f"Skipping invalid model name: {model_name}")
            continue
        
        sanitized_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in model_name)
        filename = f"{sanitized_name}.pkl"
        filepath = folder_path / filename
        
        if filepath.exists():
            logger.warning(f"Overwriting existing file: {filepath}")
        
        try:
            joblib.dump(model_object, filepath)
            logger.info(f"Saved model '{model_name}' to {filepath}")
            saved_count += 1
        except Exception as e:
            logger.error(f"Failed to save model '{model_name}': {str(e)}")
            raise Exception(f"Failed to serialize model '{model_name}': {str(e)}")
    
    if saved_count == 0:
        logger.error("No models were saved successfully")
        raise ValueError("No valid models to save")
    
    logger.info(f"Successfully saved {saved_count}/{len(models)} models")


def load_models(folder: str) -> Dict[str, Any]:
    """
    Load all models from a directory.
    
    Reads all .pkl files in the specified folder and deserializes them using
    joblib. Returns a dictionary mapping model names (derived from filenames)
    to loaded model objects.
    
    Parameters
    ----------
    folder : str
        Directory path containing saved model files (.pkl).
        
    Returns
    -------
    Dict[str, Any]
        Dictionary mapping model names to loaded model objects.
        Keys are filenames without .pkl extension.
        
    Raises
    ------
    FileNotFoundError
        If the specified folder does not exist.
    ValueError
        If folder path is invalid or no .pkl files found.
    Exception
        If deserialization fails for any model.
        
    Notes
    -----
    - Only loads files with .pkl extension
    - Model names derived from filenames (without extension)
    - Skips files that fail to load and logs errors
    - Returns empty dict if no valid models found
    """
    logger.info("Starting model load operation")
    
    if not folder or not isinstance(folder, str):
        logger.error("Invalid folder path provided")
        raise ValueError("Folder path must be a non-empty string")
    
    folder_path = Path(folder)
    
    if not folder_path.exists():
        logger.error(f"Folder does not exist: {folder_path.absolute()}")
        raise FileNotFoundError(f"Folder not found: {folder_path.absolute()}")
    
    if not folder_path.is_dir():
        logger.error(f"Path is not a directory: {folder_path.absolute()}")
        raise ValueError(f"Path is not a directory: {folder_path.absolute()}")
    
    pkl_files = list(folder_path.glob("*.pkl"))
    
    if len(pkl_files) == 0:
        logger.warning(f"No .pkl files found in {folder_path.absolute()}")
        return {}
    
    logger.info(f"Found {len(pkl_files)} .pkl files in {folder_path.absolute()}")
    
    models = {}
    loaded_count = 0
    
    for pkl_file in pkl_files:
        model_name = pkl_file.stem
        
        try:
            model_object = joblib.load(pkl_file)
            models[model_name] = model_object
            logger.info(f"Loaded model '{model_name}' from {pkl_file}")
            loaded_count += 1
        except Exception as e:
            logger.error(f"Failed to load model from {pkl_file}: {str(e)}")
            raise Exception(f"Failed to deserialize model from {pkl_file}: {str(e)}")
    
    if loaded_count == 0:
        logger.warning("No models loaded successfully")
        return {}
    
    logger.info(f"Successfully loaded {loaded_count} models")
    
    return models