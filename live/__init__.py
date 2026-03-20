"""
Live prediction module for real-time stock predictions.
"""

from .nifty50 import get_nifty50_tickers, get_ticker_name
from .predictor import get_live_predictions, fetch_live_data

__all__ = [
    'get_nifty50_tickers',
    'get_ticker_name', 
    'get_live_predictions',
    'fetch_live_data'
]
