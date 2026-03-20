import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import threading


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "experiment_logger.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class ExperimentLogger:
    """
    Thread-safe experiment tracking and logging system.
    
    Manages experiment metadata by appending results to a CSV file with
    automatic timestamping. Provides history retrieval for analysis and
    comparison of experiments over time.
    
    Attributes
    ----------
    csv_path : Path
        Path to the CSV file storing experiment logs.
    _lock : threading.Lock
        Lock for thread-safe file operations.
        
    Notes
    -----
    - All log_run operations are thread-safe via locking
    - Timestamps added automatically in ISO format
    - CSV file created on first log if it doesn't exist
    - Existing logs are never overwritten, only appended
    """
    
    def __init__(self):
        """
        Initialize ExperimentLogger with default CSV path.
        
        Creates logs directory if it doesn't exist and sets up thread lock
        for concurrent access safety.
        """
        self.csv_path = Path("logs") / "experiments.csv"
        self._lock = threading.Lock()
        
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ExperimentLogger initialized with path: {self.csv_path.absolute()}")
    
    def log_run(self, info: Dict[str, Any]) -> None:
        """
        Log experiment run information to CSV file.
        
        Appends experiment metadata to the log file with automatic timestamp.
        Creates the CSV file if it doesn't exist. Thread-safe for concurrent
        logging from multiple processes or threads.
        
        Parameters
        ----------
        info : Dict[str, Any]
            Dictionary containing experiment metadata.
            Keys become CSV columns, values must be serializable.
            Common keys: model_name, optimizer, mae, r2, sharpe, etc.
            
        Returns
        -------
        None
        
        Raises
        ------
        TypeError
            If info is not a dictionary.
        ValueError
            If info is empty.
        Exception
            If CSV write operation fails.
            
        Notes
        -----
        - Automatically adds 'timestamp' column with current datetime
        - If timestamp already in info, it will be overwritten
        - Uses thread lock to ensure atomic append operations
        - File is created with headers if it doesn't exist
        
        Examples
        --------
        >>> logger = ExperimentLogger()
        >>> logger.log_run({
        ...     'model': 'RandomForest',
        ...     'mae': 0.025,
        ...     'sharpe': 1.5
        ... })
        """
        logger.info("Starting experiment log operation")
        
        if not isinstance(info, dict):
            logger.error("Info must be a dictionary")
            raise TypeError("Info must be a dictionary")
        
        if len(info) == 0:
            logger.error("Empty info dictionary provided")
            raise ValueError("Info dictionary cannot be empty")
        
        with self._lock:
            try:
                info_copy = info.copy()
                
                info_copy['timestamp'] = datetime.now().isoformat()
                
                df_new = pd.DataFrame([info_copy])
                
                if self.csv_path.exists():
                    logger.info(f"Appending to existing log file: {self.csv_path}")
                    
                    df_existing = pd.read_csv(self.csv_path)
                    
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
                    
                    df_combined.to_csv(self.csv_path, index=False)
                    
                    logger.info(f"Appended experiment log (total rows: {len(df_combined)})")
                else:
                    logger.info(f"Creating new log file: {self.csv_path}")
                    
                    df_new.to_csv(self.csv_path, index=False)
                    
                    logger.info(f"Created new experiment log with first entry")
                
                logger.info(f"Logged experiment: {list(info.keys())}")
                
            except Exception as e:
                logger.error(f"Failed to log experiment: {str(e)}")
                raise Exception(f"Failed to write experiment log: {str(e)}")
    
    def load_history(self) -> pd.DataFrame:
        """
        Load complete experiment history from CSV file.
        
        Reads all logged experiments from the CSV file and returns as DataFrame.
        Returns empty DataFrame if no log file exists yet.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing all logged experiments with columns matching
            the keys from logged info dictionaries plus timestamp column.
            Returns empty DataFrame if file doesn't exist.
            
        Raises
        ------
        Exception
            If CSV read operation fails (file exists but is corrupted).
            
        Notes
        -----
        - Returns empty DataFrame (not None) if file doesn't exist
        - Empty DataFrame has no columns or rows
        - Does not modify the log file
        
        Examples
        --------
        >>> logger = ExperimentLogger()
        >>> history = logger.load_history()
        >>> print(f"Total experiments: {len(history)}")
        """
        logger.info("Loading experiment history")
        
        if not self.csv_path.exists():
            logger.warning(f"Log file does not exist: {self.csv_path}")
            logger.info("Returning empty DataFrame")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded experiment history: {len(df)} runs, {len(df.columns)} columns")
            
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    logger.info("Converted timestamp column to datetime")
                except Exception as e:
                    logger.warning(f"Could not convert timestamp to datetime: {str(e)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load experiment history: {str(e)}")
            raise Exception(f"Failed to read experiment log: {str(e)}")