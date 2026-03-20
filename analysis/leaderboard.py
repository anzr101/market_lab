import pandas as pd
import logging
from pathlib import Path


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "leaderboard.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def generate_leaderboard(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Generate ranked leaderboard from experiment results.
    
    Sorts experiments by specified metric and assigns ranks with best
    performance at rank 1. Automatically determines sort direction based
    on metric type (lower is better for loss metrics, higher is better
    for performance metrics).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experiment results.
        Must contain the specified metric column.
    metric : str
        Column name to sort by and rank.
        Special handling for loss metrics: 'mae', 'max_drawdown' (ascending).
        All other metrics sorted descending (higher is better).
        
    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with added 'rank' column.
        Best performing experiment has rank=1.
        Index reset to sequential integers.
        
    Raises
    ------
    ValueError
        If df is empty or metric column doesn't exist.
    TypeError
        If df is not a DataFrame or metric is not a string.
        
    Notes
    -----
    Sorting rules:
    - mae, max_drawdown: ascending (lower values ranked higher)
    - sharpe, r2, cumulative_return, etc.: descending (higher values ranked higher)
    
    Examples
    --------
    >>> results = pd.DataFrame({
    ...     'model': ['A', 'B', 'C'],
    ...     'mae': [0.05, 0.03, 0.04]
    ... })
    >>> leaderboard = generate_leaderboard(results, 'mae')
    >>> # Model B (lowest MAE) will be rank 1
    """
    logger.info("Starting leaderboard generation")
    
    if not isinstance(df, pd.DataFrame):
        logger.error("Input must be a pandas DataFrame")
        raise TypeError("df must be a pandas DataFrame")
    
    if not isinstance(metric, str):
        logger.error("Metric must be a string")
        raise TypeError("metric must be a string column name")
    
    if len(df) == 0:
        logger.error("Empty DataFrame provided")
        raise ValueError("DataFrame cannot be empty")
    
    if metric not in df.columns:
        logger.error(f"Metric column '{metric}' not found in DataFrame")
        available_columns = list(df.columns)
        logger.error(f"Available columns: {available_columns}")
        raise ValueError(f"Metric '{metric}' not found in DataFrame. Available: {available_columns}")
    
    logger.info(f"Generating leaderboard for metric: {metric}")
    logger.info(f"Input DataFrame shape: {df.shape}")
    
    df_leaderboard = df.copy()
    
    loss_metrics = {'mae', 'max_drawdown'}
    
    if metric.lower() in loss_metrics:
        ascending = True
        sort_direction = "ascending (lower is better)"
    else:
        ascending = False
        sort_direction = "descending (higher is better)"
    
    logger.info(f"Sorting {sort_direction}")
    
    df_leaderboard = df_leaderboard.sort_values(
        by=metric,
        ascending=ascending
    ).reset_index(drop=True)
    
    df_leaderboard['rank'] = range(1, len(df_leaderboard) + 1)
    
    logger.info(f"Leaderboard generated with {len(df_leaderboard)} entries")
    
    if len(df_leaderboard) > 0:
        top_value = df_leaderboard.loc[0, metric]
        logger.info(f"Top performer: rank 1 with {metric}={top_value}")
        
        if len(df_leaderboard) >= 3:
            top3_values = df_leaderboard.loc[:2, metric].tolist()
            logger.info(f"Top 3 {metric} values: {top3_values}")
    
    logger.info("Leaderboard generation complete")
    
    return df_leaderboard