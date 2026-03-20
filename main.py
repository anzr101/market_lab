import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
import os

from config.settings import Config
from data.loader import load_market_data
from features.generator import generate_features
from features.target import create_targets
from models.train import train_models
from models.save_load import save_models
from models.predict import predict
from optimizers.compare import run_optimizer_comparison
from analysis.model_analysis import analyze_model_results
from analysis.optimizer_analysis import analyze_optimizer_results
from analysis.metrics import summary_stats
from analysis.leaderboard import generate_leaderboard
from backtest.simulator import run_backtest
from utils.experiment_logger import ExperimentLogger


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "system.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def load_data(config: Config) -> pd.DataFrame:
    """
    Load and prepare dataset with features and targets.
    
    Executes the data pipeline: loads raw data, validates schema, generates
    technical features, and creates forward-looking targets. Returns a
    complete dataset ready for model training.
    
    Parameters
    ----------
    config : Config
        Configuration object containing DATA_PATH and other settings.
        
    Returns
    -------
    pd.DataFrame
        Processed dataset with OHLCV data, features, and targets.
        
    Raises
    ------
    Exception
        If any step in the data pipeline fails.
        
    Notes
    -----
    Pipeline steps:
    1. Load and validate raw CSV data
    2. Generate technical features (10 features)
    3. Create forward-looking targets (5-day returns)
    """
    logger.info("="*80)
    logger.info("STAGE 1: DATA ACQUISITION")
    logger.info("="*80)
    
    try:
        logger.info(f"Loading data from: {config.DATA_PATH}")
        df = load_market_data(config.DATA_PATH)
        logger.info(f"Data loaded successfully: {df.shape}")
        
        logger.info("="*80)
        logger.info("STAGE 2: FEATURE ENGINEERING")
        logger.info("="*80)
        
        df = generate_features(df)
        logger.info(f"Features generated: {df.shape}")
        
        logger.info("="*80)
        logger.info("STAGE 3: TARGET GENERATION")
        logger.info("="*80)
        
        df = create_targets(df)
        logger.info(f"Targets created: {df.shape}")
        
        logger.info("Data preparation complete")
        return df
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise


def train_pipeline(df: pd.DataFrame, config: Config) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    Train multiple models and save to disk.
    
    Trains LinearRegression, RandomForestRegressor, and LGBMRegressor on
    the provided dataset. Identifies best model by MAE and saves all models
    to configured directory.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with features and targets.
    config : Config
        Configuration object containing model parameters and paths.
        
    Returns
    -------
    Tuple[Dict[str, Dict[str, Any]], str]
        Tuple containing:
        - results: dictionary of model results with metrics
        - best_model_name: name of best performing model (lowest MAE)
        
    Raises
    ------
    Exception
        If training or saving fails.
        
    Notes
    -----
    - Uses chronological 80/20 train/test split
    - Evaluates models on MAE and R2
    - Best model selected by lowest MAE
    - All models saved to MODEL_DIR
    """
    logger.info("="*80)
    logger.info("STAGE 4: MODEL TRAINING")
    logger.info("="*80)
    
    try:
        logger.info("Training models")
        results = train_models(df)
        logger.info(f"Training complete: {len(results)} models trained")
        
        best_model_name = min(results.items(), key=lambda x: x[1]['mae'])[0]
        best_mae = results[best_model_name]['mae']
        logger.info(f"Best model: {best_model_name} (MAE: {best_mae:.6f})")
        
        models_to_save = {name: res['model'] for name, res in results.items()}
        
        logger.info(f"Saving models to: {config.MODEL_DIR}")
        save_models(models_to_save, config.MODEL_DIR)
        logger.info("Models saved successfully")
        
        return results, best_model_name
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


def run_optimizer_stage(data_path: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Execute optimizer comparison and analysis.
    
    Parameters
    ----------
    data_path : str
        Path to raw data file.
        
    Returns
    -------
    Tuple[Dict[str, float], Dict[str, Any]]
        Tuple containing optimizer results and analysis.
    """
    logger.info("="*80)
    logger.info("STAGE 5: OPTIMIZER COMPARISON")
    logger.info("="*80)
    
    try:
        optimizer_comparison = run_optimizer_comparison(data_path)
        optimizer_results = optimizer_comparison['optimizer_results']
        logger.info(f"Optimizers evaluated: {list(optimizer_results.keys())}")
        
        for opt_name, result in optimizer_results.items():
            logger.info(f"  {opt_name}: MAE = {result['mae']:.6f}")
            logger.info(f"    Best Params: {result['params']}")
        
        mae_only = {name: result["mae"] for name, result in optimizer_results.items()}
        optimizer_analysis = analyze_optimizer_results(mae_only)

        logger.info(f"Best optimizer: {optimizer_analysis['best_optimizer']}")
        logger.info(f"Optimizer reliability: {optimizer_analysis['reliability_grade']}")
        best_name = optimizer_analysis['best_optimizer']
        best_params = optimizer_results[best_name]['params']
        logger.info(f"Best hyperparameters: {best_params}")
        return optimizer_results, optimizer_analysis
        
    except Exception as e:
        logger.error(f"Optimizer stage failed: {str(e)}")
        raise


def run_analysis_stage(model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Execute model performance analysis.
    
    Parameters
    ----------
    model_results : Dict[str, Dict[str, Any]]
        Training results containing model metrics.
        
    Returns
    -------
    Dict[str, Any]
        Model analysis results.
    """
    logger.info("="*80)
    logger.info("STAGE 6: MODEL ANALYSIS")
    logger.info("="*80)
    
    try:
        model_metrics = {name: {'mae': res['mae'], 'r2': res['r2']} 
                        for name, res in model_results.items()}
        model_analysis = analyze_model_results(model_metrics)
        logger.info(f"Best model: {model_analysis['best_model']}")
        logger.info(f"Model reliability: {model_analysis['reliability_grade']}")
        logger.info(f"Model consistency: {model_analysis['consistency_score']:.4f}")
        
        return model_analysis
        
    except Exception as e:
        logger.error(f"Analysis stage failed: {str(e)}")
        raise


def evaluate(df: pd.DataFrame, results: Dict[str, Dict[str, Any]], best_model_name: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Generate predictions and evaluate backtest performance.
    
    Uses the best trained model to generate predictions, simulates a trading
    backtest, and computes comprehensive performance metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with features and targets.
    results : Dict[str, Dict[str, Any]]
        Training results containing trained models.
    best_model_name : str
        Name of the best performing model.
        
    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, Any]]
        Tuple containing:
        - predictions: dictionary mapping model names to prediction arrays
        - metrics: dictionary of performance metrics from backtest
        
    Raises
    ------
    Exception
        If prediction generation or backtesting fails.
        
    Notes
    -----
    - Generates predictions from all trained models
    - Runs backtest using best model predictions
    - Computes equity curve, Sharpe ratio, drawdown, win rate
    - Returns comprehensive performance metrics
    """
    logger.info("="*80)
    logger.info("STAGE 7: BACKTEST SIMULATION")
    logger.info("="*80)
    
    try:
        logger.info("Preparing features for prediction")
        df_clean = df.dropna(subset=['future_return_5d']).copy()
        
        exclude_columns = {'Date', 'Ticker', 'future_return_5d', 'direction_5d'}
        feature_columns = [col for col in df_clean.select_dtypes(include=[np.number]).columns 
                          if col not in exclude_columns]
        
        X = df_clean[feature_columns].fillna(0)
        
        logger.info("Generating predictions from all models")
        models_dict = {name: res['model'] for name, res in results.items()}
        predictions = predict(models_dict, X)
        logger.info(f"Predictions generated for {len(predictions)} models")
        
        logger.info(f"Running backtest with best model: {best_model_name}")
        best_predictions = predictions[best_model_name]
        
        backtest_results = run_backtest(df_clean, best_predictions)
        
        logger.info("="*80)
        logger.info("STAGE 8: METRICS COMPUTATION")
        logger.info("="*80)
        
        daily_returns = np.diff(backtest_results['equity_curve']) / backtest_results['equity_curve'][:-1]
        
        logger.info("Computing comprehensive metrics")
        metrics = summary_stats(daily_returns)
        
        metrics['cumulative_return'] = backtest_results['cumulative_return']
        metrics['equity_curve_final'] = backtest_results['equity_curve'][-1]
        metrics['max_drawdown'] = backtest_results['max_drawdown']
        
        logger.info(f"Sharpe Ratio: {metrics['sharpe']:.4f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        logger.info(f"Cumulative Return: {metrics['cumulative_return']*100:.2f}%")
        
        logger.info("Evaluation complete")

        
        os.makedirs("artifacts", exist_ok=True)

        logger.info("Saving prediction cache for dashboard")

        pred_df = df_clean.copy()

        for model_name, preds in predictions.items():
            pred_df[f"pred_{model_name}"] = preds

        pred_df.to_parquet("artifacts/predictions.parquet")

        logger.info("Prediction cache saved to artifacts/predictions.parquet")
        
        return predictions, metrics
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


def compute_confidence_score(
    model_analysis: Dict[str, Any],
    optimizer_analysis: Dict[str, Any],
    metrics: Dict[str, float]
) -> float:
    """
    Compute system-level confidence score for deployment readiness.
    
    Parameters
    ----------
    model_analysis : Dict[str, Any]
        Model analysis results including consistency score.
    optimizer_analysis : Dict[str, Any]
        Optimizer analysis results including stability score.
    metrics : Dict[str, float]
        Backtest performance metrics.
        
    Returns
    -------
    float
        Confidence score between 0 and 1, where 1 indicates maximum confidence.
        
    Notes
    -----
    Weighted confidence formula:
    - 30% model consistency (inverse of MAE CV)
    - 25% optimizer stability (normalized stability score)
    - 30% Sharpe quality (sigmoid normalized)
    - 15% win rate (probability measure)
    
    Higher values indicate greater system confidence in deployment recommendation.
    """
    model_consistency = model_analysis['consistency_score']
    optimizer_stability = optimizer_analysis['normalized_stability_score']
    
    sharpe = metrics.get('sharpe', 0.0)
    sharpe_sigmoid = 1.0 / (1.0 + np.exp(-sharpe / 2.0))
    
    win_rate = metrics.get('win_rate', 0.5)
    
    confidence = (
        0.30 * model_consistency +
        0.25 * optimizer_stability +
        0.30 * sharpe_sigmoid +
        0.15 * win_rate
    )
    
    confidence = np.clip(confidence, 0.0, 1.0)
    
    logger.info(f"Confidence components: model={model_consistency:.4f}, optimizer={optimizer_stability:.4f}, sharpe={sharpe_sigmoid:.4f}, win_rate={win_rate:.4f}")
    logger.info(f"System confidence score: {confidence:.4f}")
    
    return float(confidence)


def run_system() -> Dict[str, Any]:
    """
    Execute complete Market Optimization Intelligence Lab pipeline.
    
    Orchestrates the full workflow from data loading through model training,
    optimizer comparison, performance analysis, backtesting, and comprehensive
    reporting. Generates final structured report with confidence metrics.
    
    Pipeline Overview
    -----------------
    1. Data acquisition and validation
    2. Feature engineering (10 technical features)
    3. Target generation (5-day forward returns)
    4. Model training (Linear, RF, LGBM)
    5. Optimizer comparison (Grid, Random, Optuna)
    6. Performance analysis (models and optimizers)
    7. Backtest simulation
    8. Metrics computation
    9. Experiment logging
    10. Report generation
    
    Returns
    -------
    Dict[str, Any]
        Comprehensive research report containing:
        - best_model: str
        - best_optimizer: str
        - model_ranking: List[Tuple[str, float]]
        - optimizer_ranking: List[Tuple[str, float]]
        - metrics: Dict[str, float]
        - confidence_score: float
        - model_analysis: Dict[str, Any]
        - optimizer_analysis: Dict[str, Any]
        - system_metadata: Dict[str, Any]
        
    Raises
    ------
    Exception
        If any critical step in the pipeline fails.
        
    Notes
    -----
    - All operations are deterministic (fixed random seeds)
    - Results logged to experiments.csv for tracking
    - Complete audit trail maintained in log files
    - Backward compatible with existing experiment logging
    """
    logger.info("="*80)
    logger.info("MARKET OPTIMIZATION INTELLIGENCE LAB - SYSTEM STARTUP")
    logger.info("="*80)
    
    try:
        logger.info("Initializing configuration")
        config = Config()
        logger.info(f"Configuration: {config.to_dict()}")
        
        df = load_data(config)
        
        results, best_model_name = train_pipeline(df, config)
        
        optimizer_results, optimizer_analysis = run_optimizer_stage(config.DATA_PATH)
        
        model_analysis = run_analysis_stage(results)
        
        predictions, metrics = evaluate(df, results, best_model_name)
        
        logger.info("="*80)
        logger.info("STAGE 9: CONFIDENCE SCORING")
        logger.info("="*80)
        
        confidence_score = compute_confidence_score(
            model_analysis,
            optimizer_analysis,
            metrics
        )
        
        logger.info("="*80)
        logger.info("STAGE 10: EXPERIMENT LOGGING AND REPORTING")
        logger.info("="*80)
        
        exp_logger = ExperimentLogger()
        
        experiment_info = {
            'best_model': best_model_name,
            'best_optimizer': optimizer_analysis['best_optimizer'],
            'mae': results[best_model_name]['mae'],
            'r2': results[best_model_name]['r2'],
            'sharpe': metrics['sharpe'],
            'max_drawdown': metrics['max_drawdown'],
            'win_rate': metrics['win_rate'],
            'cumulative_return': metrics['cumulative_return'],
            'profit_factor': metrics['profit_factor'],
            'confidence_score': confidence_score,
            'model_consistency': model_analysis['consistency_score'],
            'optimizer_stability': optimizer_analysis['normalized_stability_score'],
            'test_size': config.TEST_SIZE,
            'random_state': config.RANDOM_STATE,
            'n_estimators': config.N_ESTIMATORS,
            'max_depth': config.MAX_DEPTH
        }
        
        logger.info("Logging experiment results")
        exp_logger.log_run(experiment_info)
        logger.info("Experiment logged successfully")
        
        logger.info("Loading experiment history")
        history = exp_logger.load_history()
        
        if len(history) > 0:
            logger.info(f"Generating leaderboard from {len(history)} experiments")
            leaderboard = generate_leaderboard(history, 'mae')
            
            logger.info("="*60)
            logger.info("TOP 5 EXPERIMENTS (BY MAE)")
            logger.info("="*60)
            
            top_5 = leaderboard.head(5)
            for idx, row in top_5.iterrows():
                logger.info(f"Rank {row['rank']}: {row['best_model']} - MAE={row['mae']:.6f}, Sharpe={row['sharpe']:.4f}")
            
            logger.info("="*60)
        else:
            logger.warning("No experiment history available for leaderboard")
        
        df_clean = df.dropna(subset=['future_return_5d']).copy()
        exclude_columns = {'Date', 'Ticker', 'future_return_5d', 'direction_5d'}
        feature_columns = [col for col in df_clean.select_dtypes(include=[np.number]).columns 
                          if col not in exclude_columns]
        
        final_report = {
            'best_model': model_analysis['best_model'],
            'best_optimizer': optimizer_analysis['best_optimizer'],
            'model_ranking': model_analysis['ranking'],
            'optimizer_ranking': optimizer_analysis['ranking'],
            'metrics': {
                'sharpe': metrics['sharpe'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'cumulative_return': metrics['cumulative_return'],
                'profit_factor': metrics['profit_factor'],
                'mean': metrics['mean'],
                'std': metrics['std']
            },
            'confidence_score': confidence_score,
            'model_analysis': {
                'performance_spread': model_analysis['performance_spread'],
                'consistency_score': model_analysis['consistency_score'],
                'reliability_grade': model_analysis['reliability_grade'],
                'recommendation': model_analysis['recommendation']
            },
            'optimizer_analysis': {
                'performance_gap': optimizer_analysis['performance_gap'],
                'stability_score': optimizer_analysis['normalized_stability_score'],
                'reliability_grade': optimizer_analysis['reliability_grade'],
                'recommendation': optimizer_analysis['recommendation']
            },
            'system_metadata': {
                'total_samples': len(df_clean),
                'feature_count': len(feature_columns),
                'models_evaluated': len(results),
                'optimizers_evaluated': len(optimizer_results),
                'test_size': config.TEST_SIZE,
                'random_state': config.RANDOM_STATE
            }, 'best_hyperparameters': optimizer_results[optimizer_analysis['best_optimizer']]['params'],
        }
        
        logger.info("="*80)
        logger.info("SYSTEM EXECUTION COMPLETE")
        logger.info("="*80)
        logger.info(f"Best Model: {final_report['best_model']}")
        logger.info(f"Best Optimizer: {final_report['best_optimizer']}")
        logger.info(f"MAE: {results[best_model_name]['mae']:.6f}")
        logger.info(f"R2: {results[best_model_name]['r2']:.6f}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe']:.4f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        logger.info(f"Cumulative Return: {metrics['cumulative_return']*100:.2f}%")
        logger.info(f"Confidence Score: {confidence_score:.4f}")
        logger.info("="*80)
        logger.info(f"Best Hyperparameters: {final_report['best_hyperparameters']}")
        
        return final_report
        
    except Exception as e:
        logger.error("="*80)
        logger.error("SYSTEM EXECUTION FAILED")
        logger.error("="*80)
        logger.error(f"Error: {str(e)}")
        logger.error("="*80)
        raise


if __name__ == "__main__":
    run_system()