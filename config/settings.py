import logging
from pathlib import Path
from typing import Dict, Any


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "config.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class Config:
    """
    Configuration management for the Market Optimization Intelligence Lab.
    
    Provides centralized configuration with validation and change tracking.
    All configuration parameters are defined as class attributes with
    defaults. Updates are logged and validated against known keys.
    
    Attributes
    ----------
    DATA_PATH : str
        Path to raw CSV data file.
    MODEL_DIR : str
        Directory for saving trained models.
    TEST_SIZE : float
        Proportion of data for test set (0.0 to 1.0).
    RANDOM_STATE : int
        Random seed for reproducibility.
    N_ESTIMATORS : int
        Number of estimators for ensemble models.
    MAX_DEPTH : int
        Maximum depth for tree-based models.
    LOG_LEVEL : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        
    Notes
    -----
    - All updates are validated and logged
    - Unknown configuration keys are rejected
    - Designed for deterministic, reproducible experiments
    """
    
    def __init__(self):
        """
        Initialize configuration with default values.
        
        Sets all configuration attributes to their default values and logs
        the initialization.
        """
        self.DATA_PATH = "data/raw.csv"
        self.MODEL_DIR = "saved_models"
        self.TEST_SIZE = 0.2
        self.RANDOM_STATE = 42
        self.N_ESTIMATORS = 100
        self.MAX_DEPTH = 10
        self.LOG_LEVEL = "INFO"
        
        logger.info("Config initialized with default values")
        logger.info(f"Configuration: {self.to_dict()}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.
        
        Returns all configuration attributes as a dictionary with attribute
        names as keys and their current values as values.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary mapping configuration parameter names to values.
            
        Notes
        -----
        Only includes public attributes (not starting with underscore).
        Excludes methods and special attributes.
        
        Examples
        --------
        >>> config = Config()
        >>> config_dict = config.to_dict()
        >>> print(config_dict['RANDOM_STATE'])
        42
        """
        config_dict = {
            'DATA_PATH': self.DATA_PATH,
            'MODEL_DIR': self.MODEL_DIR,
            'TEST_SIZE': self.TEST_SIZE,
            'RANDOM_STATE': self.RANDOM_STATE,
            'N_ESTIMATORS': self.N_ESTIMATORS,
            'MAX_DEPTH': self.MAX_DEPTH,
            'LOG_LEVEL': self.LOG_LEVEL
        }
        
        return config_dict
    
    def update(self, **kwargs) -> None:
        """
        Update configuration parameters safely.
        
        Validates that all provided keys are known configuration parameters
        and updates their values. Logs all changes for traceability.
        
        Parameters
        ----------
        **kwargs
            Keyword arguments where keys are configuration parameter names
            and values are the new values to set.
            
        Raises
        ------
        ValueError
            If any provided key is not a valid configuration parameter.
            If any value fails type or range validation.
            
        Notes
        -----
        - All changes are logged before being applied
        - Unknown keys cause immediate failure without partial updates
        - Type validation is performed for critical parameters
        - TEST_SIZE must be between 0.0 and 1.0
        - N_ESTIMATORS and MAX_DEPTH must be positive integers
        - LOG_LEVEL must be valid logging level string
        
        Examples
        --------
        >>> config = Config()
        >>> config.update(RANDOM_STATE=123, N_ESTIMATORS=200)
        >>> print(config.RANDOM_STATE)
        123
        """
        logger.info(f"Config update requested with {len(kwargs)} parameters")
        
        if len(kwargs) == 0:
            logger.warning("Empty update called, no changes made")
            return
        
        valid_keys = {
            'DATA_PATH', 'MODEL_DIR', 'TEST_SIZE', 'RANDOM_STATE',
            'N_ESTIMATORS', 'MAX_DEPTH', 'LOG_LEVEL'
        }
        
        provided_keys = set(kwargs.keys())
        unknown_keys = provided_keys - valid_keys
        
        if unknown_keys:
            logger.error(f"Unknown configuration keys: {unknown_keys}")
            raise ValueError(
                f"Unknown configuration keys: {unknown_keys}. "
                f"Valid keys are: {valid_keys}"
            )
        
        for key, new_value in kwargs.items():
            old_value = getattr(self, key)
            
            if key == 'TEST_SIZE':
                if not isinstance(new_value, (int, float)):
                    logger.error(f"TEST_SIZE must be numeric, got {type(new_value)}")
                    raise ValueError(f"TEST_SIZE must be numeric")
                if not 0.0 <= new_value <= 1.0:
                    logger.error(f"TEST_SIZE must be between 0 and 1, got {new_value}")
                    raise ValueError(f"TEST_SIZE must be between 0.0 and 1.0")
            
            if key == 'RANDOM_STATE':
                if not isinstance(new_value, int):
                    logger.error(f"RANDOM_STATE must be int, got {type(new_value)}")
                    raise ValueError(f"RANDOM_STATE must be an integer")
            
            if key == 'N_ESTIMATORS':
                if not isinstance(new_value, int):
                    logger.error(f"N_ESTIMATORS must be int, got {type(new_value)}")
                    raise ValueError(f"N_ESTIMATORS must be an integer")
                if new_value <= 0:
                    logger.error(f"N_ESTIMATORS must be positive, got {new_value}")
                    raise ValueError(f"N_ESTIMATORS must be positive")
            
            if key == 'MAX_DEPTH':
                if not isinstance(new_value, int):
                    logger.error(f"MAX_DEPTH must be int, got {type(new_value)}")
                    raise ValueError(f"MAX_DEPTH must be an integer")
                if new_value <= 0:
                    logger.error(f"MAX_DEPTH must be positive, got {new_value}")
                    raise ValueError(f"MAX_DEPTH must be positive")
            
            if key == 'LOG_LEVEL':
                valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
                if new_value not in valid_levels:
                    logger.error(f"Invalid LOG_LEVEL: {new_value}")
                    raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
            
            setattr(self, key, new_value)
            logger.info(f"Updated {key}: {old_value} -> {new_value}")
        
        logger.info("Config update complete")