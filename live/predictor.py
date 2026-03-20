"""
Live Stock Prediction Module

Fetches live/recent stock data and generates predictions using trained models.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional
import sys

# Add parent directory to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from features.generator import generate_features
from models.save_load import load_models
from models.predict import predict


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "live_predictor.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def fetch_live_data(ticker: str, days_history: int = 100) -> Optional[pd.DataFrame]:
    """
    Fetch live/recent stock data from Yahoo Finance.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'RELIANCE.NS' for NIFTY 50 stocks)
    days_history : int, default=100
        Number of days of historical data to fetch
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns [Date, Ticker, Open, High, Low, Close, Volume]
        Returns None if fetch fails
    """
    logger.info(f"Fetching live data for {ticker}, last {days_history} days")
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_history)
        
        # Fetch data from yfinance
        stock_data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        if stock_data.empty:
            logger.error(f"No data received for {ticker}")
            return None
        
        # Reset index to make Date a column
        stock_data = stock_data.reset_index()
        
        # Handle MultiIndex columns (yfinance sometimes returns this)
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.droplevel(1)
        
        # Standardize column names to Title case
        stock_data.columns = [col.title() if isinstance(col, str) else col for col in stock_data.columns]
        
        # Add Ticker column
        stock_data['Ticker'] = ticker
        
        # Select and order required columns
        required_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check if all required columns exist
        missing_cols = [col for col in required_cols if col not in stock_data.columns]
        if missing_cols:
            logger.error(f"Missing columns in data: {missing_cols}")
            logger.error(f"Available columns: {list(stock_data.columns)}")
            return None
        
        stock_data = stock_data[required_cols]
        
        # Convert Date to datetime
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        
        # Sort by date
        stock_data = stock_data.sort_values('Date').reset_index(drop=True)
        
        logger.info(f"Fetched {len(stock_data)} rows for {ticker}")
        logger.info(f"Date range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")
        
        return stock_data
        
    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def prepare_live_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for live prediction using same pipeline as training.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with generated features
    """
    logger.info("Generating features for live data")
    
    # Apply same feature engineering as training
    df_features = generate_features(df)
    
    logger.info(f"Features generated: {df_features.shape}")
    
    return df_features


def get_live_predictions(
    ticker: str,
    model_dir: str = "saved_models",
    days_history: int = 100
) -> Optional[Dict[str, Any]]:
    """
    Get live predictions for a stock ticker.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'RELIANCE.NS')
    model_dir : str, default='saved_models'
        Directory containing trained models
    days_history : int, default=100
        Number of days of historical data to fetch
        
    Returns
    -------
    dict or None
        Dictionary containing:
        {
            'ticker': str,
            'current_price': float,
            'last_date': datetime,
            'predictions': dict,  # model_name -> predicted_return
            'price_targets': dict,  # model_name -> target_price
            'consensus_return': float,
            'consensus_price': float,
            'signal': str,  # 'BUY', 'SELL', or 'HOLD'
            'confidence': float,  # 0-100
            'data_points': int
        }
        Returns None if prediction fails
    """
    logger.info(f"=" * 60)
    logger.info(f"Starting live prediction for {ticker}")
    logger.info(f"=" * 60)
    
    # Step 1: Fetch live data
    df = fetch_live_data(ticker, days_history)
    if df is None or len(df) == 0:
        logger.error(f"Failed to fetch data for {ticker}")
        return None
    
    # Step 2: Generate features
    df_features = prepare_live_features(df)
    
    # Step 3: Load trained models
    logger.info(f"Loading models from {model_dir}")
    try:
        models = load_models(model_dir)
        if len(models) == 0:
            logger.error("No models found")
            return None
        logger.info(f"Loaded {len(models)} models: {list(models.keys())}")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        return None
    
    # Step 4: Prepare features for prediction (last row only)
    # Get feature columns (exclude Date, Ticker, target columns)
    exclude_columns = {'Date', 'Ticker', 'future_return_5d', 'direction_5d'}
    feature_columns = [col for col in df_features.select_dtypes(include=[np.number]).columns 
                      if col not in exclude_columns]
    
    X = df_features[feature_columns].fillna(0)
    
    # Step 5: Generate predictions
    logger.info("Generating predictions from all models")
    try:
        all_predictions = predict(models, X)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return None
    
    # Step 6: Extract latest prediction for each model
    current_price = float(df_features['Close'].iloc[-1])
    last_date = df_features['Date'].iloc[-1]
    
    model_predictions = {}
    price_targets = {}
    valid_predictions = []
    
    for model_name, preds in all_predictions.items():
        # Get last valid prediction
        latest_pred = preds[-1]
        
        if not np.isnan(latest_pred):
            model_predictions[model_name] = float(latest_pred)
            # Calculate price target: current_price * exp(predicted_return)
            target_price = current_price * np.exp(latest_pred)
            price_targets[model_name] = float(target_price)
            valid_predictions.append(latest_pred)
            
            logger.info(f"{model_name}: Return={latest_pred:.6f}, Target=₹{target_price:.2f}")
        else:
            logger.warning(f"{model_name}: NaN prediction")
    
    if len(valid_predictions) == 0:
        logger.error("All predictions are NaN")
        return None
    
    # Step 7: Calculate consensus
    consensus_return = float(np.mean(valid_predictions))
    consensus_price = current_price * np.exp(consensus_return)
    
    # Step 8: Determine signal
    if consensus_return > 0.01:  # >1% expected return
        signal = "BUY"
    elif consensus_return < -0.01:  # <-1% expected return
        signal = "SELL"
    else:
        signal = "HOLD"
    
    # Step 9: Calculate confidence (0-100)
    # Based on agreement between models
    if len(valid_predictions) > 1:
        prediction_std = np.std(valid_predictions)
        # Lower std = higher confidence
        confidence = float(100 * (1 - min(prediction_std * 10, 1)))
    else:
        confidence = 50.0
    
    result = {
        'ticker': ticker,
        'current_price': current_price,
        'last_date': last_date,
        'predictions': model_predictions,
        'price_targets': price_targets,
        'consensus_return': consensus_return,
        'consensus_price': consensus_price,
        'signal': signal,
        'confidence': confidence,
        'data_points': len(df),
        'num_models': len(model_predictions)
    }
    
    logger.info(f"Consensus: Return={consensus_return:.6f}, Price=₹{consensus_price:.2f}")
    logger.info(f"Signal: {signal}, Confidence: {confidence:.1f}%")
    logger.info("Live prediction complete")
    
    return result


if __name__ == "__main__":
    # Test with a NIFTY 50 stock
    result = get_live_predictions("RELIANCE.NS")
    
    if result:
        print(f"\n{'='*60}")
        print(f"LIVE PREDICTION: {result['ticker']}")
        print(f"{'='*60}")
        print(f"Current Price: ₹{result['current_price']:.2f}")
        print(f"Last Update: {result['last_date']}")
        print(f"\nPredictions (5-day forward return):")
        for model, pred_return in result['predictions'].items():
            target = result['price_targets'][model]
            print(f"  {model:20s}: {pred_return:+.4f} → ₹{target:.2f}")
        print(f"\nConsensus Prediction:")
        print(f"  Expected Return: {result['consensus_return']:+.4f}")
        print(f"  Target Price: ₹{result['consensus_price']:.2f}")
        print(f"  Signal: {result['signal']}")
        print(f"  Confidence: {result['confidence']:.1f}%")
        print(f"{'='*60}\n")