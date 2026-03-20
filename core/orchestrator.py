import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any

from data.loader import load_market_data
from features.generator import generate_features
from features.target import create_targets
from models.train import train_models
from optimizers.compare import run_optimizer_comparison
from analysis.optimizer_analysis import analyze_optimizer_results
from analysis.model_analysis import analyze_model_results
from backtest.simulator import run_backtest
from analysis.metrics import summary_stats


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "orchestrator.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def run_full_pipeline(data_path: str) -> Dict[str, Any]:
    """
    Execute complete Market Optimization Intelligence Lab pipeline.
    
    Orchestrates end-to-end workflow from data loading through model training,
    optimizer comparison, performance analysis, and final report generation.
    Integrates all system modules into unified research execution.
    
    Parameters
    ----------
    data_path : str
        Path to raw CSV data file containing OHLCV market data.
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive research report containing:
        - best_model: str, optimal model architecture by MAE
        - best_optimizer: str, optimal hyperparameter optimizer by MAE
        - model_ranking: List[Tuple[str, float]], models sorted by MAE
        - optimizer_ranking: List[Tuple[str, float]], optimizers sorted by MAE
        - metrics: Dict[str, float], backtest performance metrics
        - confidence_score: float, system confidence in results (0-1)
        - model_analysis: Dict[str, Any], detailed model comparison
        - optimizer_analysis: Dict[str, Any], detailed optimizer comparison
        - system_metadata: Dict[str, Any], pipeline execution metadata
        
    Raises
    ------
    ValueError
        If data_path is invalid or pipeline encounters irrecoverable error.
    FileNotFoundError
        If data file does not exist.
    Exception
        If any critical pipeline stage fails.
        
    Notes
    -----
    Pipeline stages:
    1. Data loading and validation
    2. Feature engineering (10 technical features)
    3. Target generation (5-day forward returns)
    4. Model training (Linear, RF, LGBM)
    5. Optimizer comparison (Grid, Random, Optuna)
    6. Performance analysis and ranking
    7. Backtest simulation
    8. Metrics computation
    9. Confidence scoring
    
    All operations are deterministic with fixed random states.
    Complete execution trace logged for audit and reproducibility.
    """
    logger.info("="*80)
    logger.info("MARKET OPTIMIZATION INTELLIGENCE LAB - PIPELINE ORCHESTRATION")
    logger.info("="*80)
    
    if not data_path or not isinstance(data_path, str):
        logger.error("Invalid data_path provided")
        raise ValueError("data_path must be a non-empty string")
    
    logger.info(f"Pipeline input: {data_path}")
    
    try:
        logger.info("STAGE 1: DATA ACQUISITION")
        logger.info("-" * 80)
        df = load_market_data(data_path)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        logger.info("STAGE 2: FEATURE ENGINEERING")
        logger.info("-" * 80)
        df = generate_features(df)
        logger.info(f"Features generated: {df.shape[1]} total columns")
        
        logger.info("STAGE 3: TARGET GENERATION")
        logger.info("-" * 80)
        df = create_targets(df)
        logger.info(f"Targets created: {df.shape}")
        
        logger.info("STAGE 4: MODEL TRAINING")
        logger.info("-" * 80)
        model_results = train_models(df)
        logger.info(f"Models trained: {list(model_results.keys())}")
        
        logger.info("STAGE 5: OPTIMIZER COMPARISON")
        logger.info("-" * 80)
        optimizer_comparison = run_optimizer_comparison(data_path)
        optimizer_results = optimizer_comparison['optimizer_results']
        logger.info(f"Optimizers evaluated: {list(optimizer_results.keys())}")
        
        logger.info("STAGE 6: OPTIMIZER ANALYSIS")
        logger.info("-" * 80)
        optimizer_analysis = analyze_optimizer_results(optimizer_results)
        logger.info(f"Best optimizer: {optimizer_analysis['best_optimizer']}")
        logger.info(f"Optimizer reliability: {optimizer_analysis['reliability_grade']}")
        
        logger.info("STAGE 7: MODEL ANALYSIS")
        logger.info("-" * 80)
        model_metrics = {name: {'mae': res['mae'], 'r2': res['r2']} 
                        for name, res in model_results.items()}
        model_analysis = analyze_model_results(model_metrics)
        logger.info(f"Best model: {model_analysis['best_model']}")
        logger.info(f"Model reliability: {model_analysis['reliability_grade']}")
        
        logger.info("STAGE 8: BACKTEST SIMULATION")
        logger.info("-" * 80)
        best_model_name = model_analysis['best_model']
        best_model = model_results[best_model_name]['model']
        
        df_clean = df.dropna(subset=['future_return_5d']).copy()
        exclude_columns = {'Date', 'Ticker', 'future_return_5d', 'direction_5d'}
        feature_columns = [col for col in df_clean.select_dtypes(include=[np.number]).columns 
                          if col not in exclude_columns]
        X = df_clean[feature_columns].fillna(0)
        
        predictions = best_model.predict(X)
        backtest_results = run_backtest(df_clean, predictions)
        
        logger.info(f"Backtest complete: Sharpe={backtest_results.get('sharpe', 0):.4f}")
        
        logger.info("STAGE 9: METRICS COMPUTATION")
        logger.info("-" * 80)
        daily_returns = np.diff(backtest_results['equity_curve']) / backtest_results['equity_curve'][:-1]
        metrics = summary_stats(daily_returns)
        metrics['cumulative_return'] = backtest_results['cumulative_return']
        metrics['max_drawdown'] = backtest_results['max_drawdown']
        
        logger.info(f"Sharpe: {metrics['sharpe']:.4f}")
        logger.info(f"Max DD: {metrics['max_drawdown']*100:.2f}%")
        logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        
        logger.info("STAGE 10: CONFIDENCE SCORING")
        logger.info("-" * 80)
        confidence_score = _compute_confidence_score(
            model_analysis,
            optimizer_analysis,
            metrics
        )
        logger.info(f"System confidence: {confidence_score:.4f}")
        
        logger.info("STAGE 11: REPORT GENERATION")
        logger.info("-" * 80)
        
        final_report = {
            'best_model': model_analysis['best_model'],
            'best_optimizer': optimizer_analysis['best_optimizer'],
            'model_ranking': model_analysis['ranking'],
            'optimizer_ranking': optimizer_analysis['ranking'],
            'metrics': metrics,
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
                'train_size': optimizer_comparison['train_size'],
                'test_size': optimizer_comparison['test_size'],
                'models_evaluated': len(model_results),
                'optimizers_evaluated': len(optimizer_results)
            }
        }
        
        logger.info("="*80)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("="*80)
        logger.info(f"Best Model: {final_report['best_model']}")
        logger.info(f"Best Optimizer: {final_report['best_optimizer']}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe']:.4f}")
        logger.info(f"Confidence: {confidence_score:.4f}")
        logger.info("="*80)
        
        return final_report
        
    except Exception as e:
        logger.error("="*80)
        logger.error("PIPELINE EXECUTION FAILED")
        logger.error("="*80)
        logger.error(f"Error: {str(e)}")
        logger.error("="*80)
        raise


def _compute_confidence_score(
    model_analysis: Dict[str, Any],
    optimizer_analysis: Dict[str, Any],
    metrics: Dict[str, float]
) -> float:
    """
    Compute system-level confidence score for results.
    
    Parameters
    ----------
    model_analysis : Dict[str, Any]
        Model analysis results including consistency and spread.
    optimizer_analysis : Dict[str, Any]
        Optimizer analysis results including stability.
    metrics : Dict[str, float]
        Backtest performance metrics.
        
    Returns
    -------
    float
        Confidence score between 0 and 1, where 1 indicates maximum confidence.
        
    Notes
    -----
    Confidence synthesizes:
    - Model consistency (inverse of MAE coefficient of variation)
    - Optimizer stability (normalized stability score)
    - Predictive quality (Sharpe ratio, bounded and normalized)
    - Statistical reliability (win rate as probability measure)
    
    Higher consistency, stability, and performance yield higher confidence.
    Score represents system's certainty in deployment recommendation.
    """
    model_consistency = model_analysis['consistency_score']
    optimizer_stability = optimizer_analysis['normalized_stability_score']
    
    sharpe = metrics.get('sharpe', 0.0)
    sharpe_component = 1.0 / (1.0 + np.exp(-sharpe / 2.0))
    
    win_rate = metrics.get('win_rate', 0.5)
    win_rate_component = win_rate
    
    weights = {
        'model_consistency': 0.30,
        'optimizer_stability': 0.25,
        'sharpe_quality': 0.30,
        'win_rate': 0.15
    }
    
    confidence = (
        weights['model_consistency'] * model_consistency +
        weights['optimizer_stability'] * optimizer_stability +
        weights['sharpe_quality'] * sharpe_component +
        weights['win_rate'] * win_rate_component
    )
    
    confidence = np.clip(confidence, 0.0, 1.0)
    
    return float(confidence)