import pandas as pd
import numpy as np
import logging
from pathlib import Path


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "feature_generator.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive technical features for time-series modeling.
    
    MASSIVELY UPGRADED VERSION with 70+ features including:
    - Volume analysis (OBV, volume momentum, breakouts)
    - Momentum indicators (RSI, MACD, ROC, Williams %R)
    - Volatility measures (ATR, Bollinger Bands, regime detection)
    - Trend analysis (multiple MAs, ADX approximation)
    - Price action (gaps, ranges, candle patterns)
    - Lagged features (historical returns and volume)
    - Cross-sectional features (ranks, z-scores)
    
    All features maintain strict time-series safety (no lookahead bias).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns: Date, Ticker, Open, High, Low, Close, Volume.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus ~70 new feature columns.
    """
    logger.info("Starting ENHANCED feature generation with 70+ features")
    
    required_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    df_out = df.copy()
    initial_rows = len(df_out)
    logger.info(f"Processing {initial_rows} rows")
    
    if not pd.api.types.is_datetime64_any_dtype(df_out['Date']):
        logger.warning("Date column is not datetime type, attempting conversion")
        df_out['Date'] = pd.to_datetime(df_out['Date'])
    
    df_out = df_out.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # ==================== BASIC FEATURES ====================
    logger.info("Computing basic features")
    
    # Returns
    df_out['log_return'] = df_out.groupby('Ticker')['Close'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df_out['simple_return'] = df_out.groupby('Ticker')['Close'].transform(
        lambda x: x.pct_change()
    )
    
    # ==================== LAGGED FEATURES (CRITICAL!) ====================
    logger.info("Computing lagged features")
    
    for lag in [1, 2, 3, 5, 10]:
        df_out[f'return_lag_{lag}'] = df_out.groupby('Ticker')['log_return'].shift(lag)
        df_out[f'volume_lag_{lag}'] = df_out.groupby('Ticker')['Volume'].shift(lag)
    
    # ==================== VOLUME FEATURES (THE GAME CHANGER!) ====================
    logger.info("Computing volume features")
    
    # Volume moving averages
    df_out['volume_ma_5'] = df_out.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(5, min_periods=5).mean()
    )
    df_out['volume_ma_10'] = df_out.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(10, min_periods=10).mean()
    )
    df_out['volume_ma_20'] = df_out.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )
    
    # Volume ratios (critical for breakouts)
    df_out['volume_ratio_5'] = np.where(
        df_out['volume_ma_5'] != 0,
        df_out['Volume'] / df_out['volume_ma_5'],
        np.nan
    )
    df_out['volume_ratio_20'] = np.where(
        df_out['volume_ma_20'] != 0,
        df_out['Volume'] / df_out['volume_ma_20'],
        np.nan
    )
    
    # On-Balance Volume (OBV) - VERY POWERFUL
    def calculate_obv(close, volume):
        obv = np.zeros(len(close))
        obv[0] = volume.iloc[0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv[i] = obv[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv[i] = obv[i-1] - volume.iloc[i]
            else:
                obv[i] = obv[i-1]
        return obv
    
    df_out['obv'] = df_out.groupby('Ticker', group_keys=False)[['Close', 'Volume']].apply(
        lambda g: pd.Series(calculate_obv(g['Close'], g['Volume']), index=g.index)
    ).values
    df_out['obv_ma_10'] = df_out.groupby('Ticker')['obv'].transform(
        lambda x: x.rolling(10, min_periods=10).mean()
    )
    df_out['obv_momentum'] = df_out['obv'] - df_out['obv_ma_10']
    
    # Volume momentum
    df_out['volume_momentum'] = df_out.groupby('Ticker')['Volume'].transform(
        lambda x: x - x.shift(5)
    )
    
    # Price-Volume correlation
    df_out['price_volume_corr'] = df_out.groupby('Ticker', group_keys=False)[['Close', 'Volume']].apply(
        lambda g: g['Close'].rolling(20, min_periods=20).corr(g['Volume'])
    ).values
    
    # Volume z-score
    volume_rolling_mean = df_out.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )
    volume_rolling_std = df_out.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(20, min_periods=20).std()
    )
    df_out['volume_zscore'] = np.where(
        volume_rolling_std != 0,
        (df_out['Volume'] - volume_rolling_mean) / volume_rolling_std,
        np.nan
    )
    
    # ==================== VOLATILITY FEATURES (CRITICAL!) ====================
    logger.info("Computing volatility features")
    
    # True Range and ATR
    df_out['high_low'] = df_out['High'] - df_out['Low']
    df_out['high_close'] = abs(df_out['High'] - df_out.groupby('Ticker')['Close'].shift(1))
    df_out['low_close'] = abs(df_out['Low'] - df_out.groupby('Ticker')['Close'].shift(1))
    df_out['true_range'] = df_out[['high_low', 'high_close', 'low_close']].max(axis=1)
    
    df_out['atr_14'] = df_out.groupby('Ticker')['true_range'].transform(
        lambda x: x.rolling(14, min_periods=14).mean()
    )
    df_out['atr_7'] = df_out.groupby('Ticker')['true_range'].transform(
        lambda x: x.rolling(7, min_periods=7).mean()
    )
    
    # Normalized ATR
    df_out['atr_pct'] = np.where(
        df_out['Close'] != 0,
        df_out['atr_14'] / df_out['Close'],
        np.nan
    )
    
    # Historical Volatility (multiple windows)
    df_out['volatility_5'] = df_out.groupby('Ticker')['log_return'].transform(
        lambda x: x.rolling(5, min_periods=5).std()
    )
    df_out['volatility_10'] = df_out.groupby('Ticker')['log_return'].transform(
        lambda x: x.rolling(10, min_periods=10).std()
    )
    df_out['volatility_20'] = df_out.groupby('Ticker')['log_return'].transform(
        lambda x: x.rolling(20, min_periods=20).std()
    )
    
    # Volatility ratios
    df_out['volatility_ratio_10_20'] = np.where(
        df_out['volatility_20'] != 0,
        df_out['volatility_10'] / df_out['volatility_20'],
        np.nan
    )
    
    # Bollinger Bands
    df_out['bb_middle'] = df_out.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )
    bb_std = df_out.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(20, min_periods=20).std()
    )
    df_out['bb_upper'] = df_out['bb_middle'] + (2 * bb_std)
    df_out['bb_lower'] = df_out['bb_middle'] - (2 * bb_std)
    
    # Bollinger Band position and width
    df_out['bb_position'] = np.where(
        (df_out['bb_upper'] - df_out['bb_lower']) != 0,
        (df_out['Close'] - df_out['bb_lower']) / (df_out['bb_upper'] - df_out['bb_lower']),
        np.nan
    )
    df_out['bb_width'] = np.where(
        df_out['bb_middle'] != 0,
        (df_out['bb_upper'] - df_out['bb_lower']) / df_out['bb_middle'],
        np.nan
    )
    
    # Volatility regime (expanding volatility detection)
    vol_ma_20 = df_out.groupby('Ticker')['volatility_5'].transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )
    df_out['volatility_regime'] = np.where(
        vol_ma_20 != 0,
        df_out['volatility_5'] / vol_ma_20,
        np.nan
    )
    
    # ==================== MOMENTUM INDICATORS (ESSENTIAL!) ====================
    logger.info("Computing momentum indicators")
    
    # RSI (Relative Strength Index) - 14 period
    def calculate_rsi(group, period=14):
        delta = group.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df_out['rsi_14'] = df_out.groupby('Ticker')['Close'].transform(
        lambda x: calculate_rsi(x, 14)
    )
    df_out['rsi_7'] = df_out.groupby('Ticker')['Close'].transform(
        lambda x: calculate_rsi(x, 7)
    )
    
    # MACD
    ema_12 = df_out.groupby('Ticker')['Close'].transform(
        lambda x: x.ewm(span=12, adjust=False, min_periods=12).mean()
    )
    ema_26 = df_out.groupby('Ticker')['Close'].transform(
        lambda x: x.ewm(span=26, adjust=False, min_periods=26).mean()
    )
    df_out['macd'] = ema_12 - ema_26
    df_out['macd_signal'] = df_out.groupby('Ticker')['macd'].transform(
        lambda x: x.ewm(span=9, adjust=False, min_periods=9).mean()
    )
    df_out['macd_histogram'] = df_out['macd'] - df_out['macd_signal']
    
    # Rate of Change (ROC)
    df_out['roc_5'] = df_out.groupby('Ticker')['Close'].transform(
        lambda x: (x - x.shift(5)) / x.shift(5)
    )
    df_out['roc_10'] = df_out.groupby('Ticker')['Close'].transform(
        lambda x: (x - x.shift(10)) / x.shift(10)
    )
    df_out['roc_20'] = df_out.groupby('Ticker')['Close'].transform(
        lambda x: (x - x.shift(20)) / x.shift(20)
    )
    
    # Momentum (simple)
    df_out['momentum_5'] = df_out.groupby('Ticker')['Close'].transform(
        lambda x: x - x.shift(5)
    )
    df_out['momentum_10'] = df_out.groupby('Ticker')['Close'].transform(
        lambda x: x - x.shift(10)
    )
    
    # Williams %R
    high_14 = df_out.groupby('Ticker')['High'].transform(
        lambda x: x.rolling(14, min_periods=14).max()
    )
    low_14 = df_out.groupby('Ticker')['Low'].transform(
        lambda x: x.rolling(14, min_periods=14).min()
    )
    df_out['williams_r'] = np.where(
        (high_14 - low_14) != 0,
        -100 * (high_14 - df_out['Close']) / (high_14 - low_14),
        np.nan
    )
    
    # ==================== TREND FEATURES ====================
    logger.info("Computing trend features")
    
    # Multiple Moving Averages
    df_out['ma_5'] = df_out.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(5, min_periods=5).mean()
    )
    df_out['ma_10'] = df_out.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(10, min_periods=10).mean()
    )
    df_out['ma_20'] = df_out.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )
    df_out['ma_50'] = df_out.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(50, min_periods=50).mean()
    )
    
    # Distance from MAs (normalized)
    df_out['ma_distance_5'] = np.where(
        df_out['ma_5'] != 0,
        (df_out['Close'] - df_out['ma_5']) / df_out['ma_5'],
        np.nan
    )
    df_out['ma_distance_10'] = np.where(
        df_out['ma_10'] != 0,
        (df_out['Close'] - df_out['ma_10']) / df_out['ma_10'],
        np.nan
    )
    df_out['ma_distance_20'] = np.where(
        df_out['ma_20'] != 0,
        (df_out['Close'] - df_out['ma_20']) / df_out['ma_20'],
        np.nan
    )
    df_out['ma_distance_50'] = np.where(
        df_out['ma_50'] != 0,
        (df_out['Close'] - df_out['ma_50']) / df_out['ma_50'],
        np.nan
    )
    
    # MA Crossovers
    df_out['ma_cross_5_20'] = np.where(
        df_out['ma_20'] != 0,
        (df_out['ma_5'] - df_out['ma_20']) / df_out['ma_20'],
        np.nan
    )
    df_out['ma_cross_10_50'] = np.where(
        df_out['ma_50'] != 0,
        (df_out['ma_10'] - df_out['ma_50']) / df_out['ma_50'],
        np.nan
    )
    
    # MA slopes
    df_out['ma_slope_5'] = df_out.groupby('Ticker')['ma_5'].transform(
        lambda x: x - x.shift(3)
    )
    df_out['ma_slope_20'] = df_out.groupby('Ticker')['ma_20'].transform(
        lambda x: x - x.shift(5)
    )
    
    # ADX Approximation (trend strength) - Simplified version
    high_shift = df_out.groupby('Ticker')['High'].shift(1)
    low_shift = df_out.groupby('Ticker')['Low'].shift(1)
    
    up_move = df_out['High'] - high_shift
    down_move = low_shift - df_out['Low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm_smooth = df_out.groupby('Ticker')['true_range'].transform(
        lambda x: pd.Series(plus_dm[x.index]).rolling(14, min_periods=14).mean()
    )
    minus_dm_smooth = df_out.groupby('Ticker')['true_range'].transform(
        lambda x: pd.Series(minus_dm[x.index]).rolling(14, min_periods=14).mean()
    )
    tr_smooth = df_out.groupby('Ticker')['true_range'].transform(
        lambda x: x.rolling(14, min_periods=14).mean()
    )
    
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, np.nan)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, np.nan)
    
    di_sum = plus_di + minus_di
    dx = np.where(di_sum != 0, 100 * np.abs(plus_di - minus_di) / di_sum, np.nan)
    
    df_out['adx_approx'] = df_out.groupby('Ticker')['true_range'].transform(
        lambda x: pd.Series(dx[x.index]).rolling(14, min_periods=14).mean()
    )
    
    # Trend regime (price above/below MA50)
    df_out['trend_regime'] = (df_out['Close'] > df_out['ma_50']).astype(float)
    
    # ==================== PRICE ACTION FEATURES ====================
    logger.info("Computing price action features")
    
    # Daily range (normalized)
    df_out['price_range'] = np.where(
        df_out['Close'] != 0,
        (df_out['High'] - df_out['Low']) / df_out['Close'],
        np.nan
    )
    
    # Close position in daily range
    df_out['close_position'] = np.where(
        (df_out['High'] - df_out['Low']) != 0,
        (df_out['Close'] - df_out['Low']) / (df_out['High'] - df_out['Low']),
        np.nan
    )
    
    # Gap detection
    prev_close = df_out.groupby('Ticker')['Close'].shift(1)
    df_out['gap'] = np.where(
        prev_close != 0,
        (df_out['Open'] - prev_close) / prev_close,
        np.nan
    )
    df_out['gap_up'] = (df_out['gap'] > 0).astype(float)
    df_out['gap_down'] = (df_out['gap'] < 0).astype(float)
    
    # Candle body and shadows
    df_out['body'] = abs(df_out['Close'] - df_out['Open'])
    df_out['upper_shadow'] = df_out['High'] - df_out[['Close', 'Open']].max(axis=1)
    df_out['lower_shadow'] = df_out[['Close', 'Open']].min(axis=1) - df_out['Low']
    
    # Body/shadow ratios
    df_out['body_to_range'] = np.where(
        df_out['price_range'] != 0,
        df_out['body'] / (df_out['High'] - df_out['Low']),
        np.nan
    )
    
    # ==================== CROSS-SECTIONAL FEATURES ====================
    logger.info("Computing cross-sectional features")
    
    # Daily rank
    df_out['daily_rank'] = df_out.groupby('Date')['log_return'].rank(
        method='average', na_option='keep'
    )
    
    # Return z-score within date
    df_out['return_zscore'] = df_out.groupby('Date')['log_return'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    # Volume rank within date
    df_out['volume_rank'] = df_out.groupby('Date')['Volume'].rank(
        method='average', na_option='keep'
    )
    
    # ==================== CLEAN UP ====================
    # Drop intermediate columns
    cols_to_drop = ['high_low', 'high_close', 'low_close', 'body', 'upper_shadow', 'lower_shadow']
    df_out = df_out.drop(columns=[col for col in cols_to_drop if col in df_out.columns])
    
    df_out = df_out.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    # Log feature statistics
    feature_columns = [col for col in df_out.columns if col not in required_columns]
    logger.info(f"Generated {len(feature_columns)} features")
    
    for col in feature_columns[:20]:  # Log first 20 for brevity
        nan_count = df_out[col].isna().sum()
        nan_pct = 100 * nan_count / len(df_out)
        logger.info(f"Feature '{col}': {nan_count} NaN values ({nan_pct:.2f}%)")
    
    logger.info(f"ENHANCED feature generation complete. Output shape: {df_out.shape}")
    logger.info(f"Total features created: {len(feature_columns)}")
    
    return df_out
