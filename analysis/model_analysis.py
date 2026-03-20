import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "model_analysis.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def analyze_model_results(results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Analyze trained model performance and generate research insights.
    
    Evaluates model comparison results based on MAE and R2 metrics, computing
    rankings, consistency scores, and statistical distributions. Produces
    deployment recommendations and reliability assessments.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, float]]
        Nested dictionary mapping model names to performance metrics.
        Each model must contain 'mae' and 'r2' keys with numeric values.
        Lower MAE indicates better performance.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - best_model: str, name of best performing model (lowest MAE)
        - worst_model: str, name of worst performing model (highest MAE)
        - ranking: List[Tuple[str, float]], sorted list of (name, mae)
        - performance_spread: Dict[str, float], MAE and R2 spread metrics
        - consistency_score: float, inverse CV score (0-1, higher is better)
        - statistical_summary: Dict[str, Dict], mean, std, cv for MAE and R2
        - recommendation: str, deployment recommendation text
        - reliability_grade: str, grade from A+ to F
        
    Raises
    ------
    TypeError
        If results structure is invalid or contains non-numeric values.
    ValueError
        If results is empty, missing required metrics, or contains invalid values.
        
    Notes
    -----
    - Primary ranking criterion is MAE (lower is better)
    - R2 values should be between -inf and 1.0 for valid linear models
    - Consistency score based on MAE coefficient of variation
    - Performance spread includes both absolute and relative measures
    - Reliability grade synthesizes performance gap and consistency
    """
    logger.info("="*60)
    logger.info("MODEL PERFORMANCE ANALYSIS INITIATED")
    logger.info("="*60)
    
    if not isinstance(results, dict):
        logger.error("Invalid input type: results must be dictionary")
        raise TypeError("results must be a dictionary mapping model names to metrics")
    
    if len(results) == 0:
        logger.error("Empty results dictionary provided")
        raise ValueError("results dictionary cannot be empty")
    
    logger.info(f"Analyzing {len(results)} model results")
    
    for model_name, metrics in results.items():
        if not isinstance(model_name, str):
            logger.error(f"Invalid model name type: {type(model_name)}")
            raise TypeError(f"Model name must be string, received {type(model_name)}")
        
        if not isinstance(metrics, dict):
            logger.error(f"Invalid metrics type for {model_name}: {type(metrics)}")
            raise TypeError(f"Metrics for '{model_name}' must be dictionary")
        
        if 'mae' not in metrics:
            logger.error(f"Missing 'mae' key for {model_name}")
            raise ValueError(f"Model '{model_name}' missing required 'mae' metric")
        
        if 'r2' not in metrics:
            logger.error(f"Missing 'r2' key for {model_name}")
            raise ValueError(f"Model '{model_name}' missing required 'r2' metric")
        
        mae = metrics['mae']
        r2 = metrics['r2']
        
        if not isinstance(mae, (int, float)):
            logger.error(f"Invalid MAE type for {model_name}: {type(mae)}")
            raise ValueError(f"MAE for '{model_name}' must be numeric, received {type(mae)}")
        
        if not isinstance(r2, (int, float)):
            logger.error(f"Invalid R2 type for {model_name}: {type(r2)}")
            raise ValueError(f"R2 for '{model_name}' must be numeric, received {type(r2)}")
        
        if not np.isfinite(mae):
            logger.error(f"Non-finite MAE value for {model_name}: {mae}")
            raise ValueError(f"MAE for '{model_name}' must be finite")
        
        if not np.isfinite(r2):
            logger.error(f"Non-finite R2 value for {model_name}: {r2}")
            raise ValueError(f"R2 for '{model_name}' must be finite")
        
        if mae < 0:
            logger.error(f"Negative MAE value for {model_name}: {mae}")
            raise ValueError(f"MAE for '{model_name}' cannot be negative")
        
        logger.info(f"  {model_name}: MAE={mae:.6f}, R2={r2:.6f}")
    
    best_model = min(results.items(), key=lambda x: x[1]['mae'])
    worst_model = max(results.items(), key=lambda x: x[1]['mae'])
    
    logger.info(f"Best performer: {best_model[0]} (MAE: {best_model[1]['mae']:.6f}, R2: {best_model[1]['r2']:.6f})")
    logger.info(f"Worst performer: {worst_model[0]} (MAE: {worst_model[1]['mae']:.6f}, R2: {worst_model[1]['r2']:.6f})")
    
    ranking = sorted(results.items(), key=lambda x: x[1]['mae'])
    logger.info(f"Ranking order: {[name for name, _ in ranking]}")
    
    mae_values = np.array([metrics['mae'] for metrics in results.values()])
    r2_values = np.array([metrics['r2'] for metrics in results.values()])
    
    mae_mean = float(np.mean(mae_values))
    mae_std = float(np.std(mae_values, ddof=1 if len(mae_values) > 1 else 0))
    mae_cv = mae_std / mae_mean if mae_mean > 0 else 0.0
    mae_range = float(np.ptp(mae_values))
    
    r2_mean = float(np.mean(r2_values))
    r2_std = float(np.std(r2_values, ddof=1 if len(r2_values) > 1 else 0))
    r2_cv = abs(r2_std / r2_mean) if r2_mean != 0 else float('inf')
    r2_range = float(np.ptp(r2_values))
    
    statistical_summary = {
        'mae': {
            'mean': mae_mean,
            'std': mae_std,
            'cv': mae_cv,
            'range': mae_range,
            'min': float(np.min(mae_values)),
            'max': float(np.max(mae_values))
        },
        'r2': {
            'mean': r2_mean,
            'std': r2_std,
            'cv': r2_cv,
            'range': r2_range,
            'min': float(np.min(r2_values)),
            'max': float(np.max(r2_values))
        }
    }
    
    logger.info(f"MAE statistics: mean={mae_mean:.6f}, std={mae_std:.6f}, cv={mae_cv:.4f}")
    logger.info(f"R2 statistics: mean={r2_mean:.6f}, std={r2_std:.6f}, cv={r2_cv:.4f}")
    
    performance_spread = {
        'mae_gap': worst_model[1]['mae'] - best_model[1]['mae'],
        'mae_relative_gap': (worst_model[1]['mae'] - best_model[1]['mae']) / mae_mean if mae_mean > 0 else 0.0,
        'r2_gap': best_model[1]['r2'] - worst_model[1]['r2'],
        'r2_relative_gap': abs(best_model[1]['r2'] - worst_model[1]['r2']) / abs(r2_mean) if r2_mean != 0 else 0.0
    }
    
    logger.info(f"Performance spread: MAE gap={performance_spread['mae_gap']:.6f}, R2 gap={performance_spread['r2_gap']:.6f}")
    
    if len(results) == 1:
        consistency_score = 1.0
        logger.info("Single model detected, consistency score set to 1.0")
    else:
        consistency_score = 1.0 / (1.0 + mae_cv)
        logger.info(f"Consistency score: {consistency_score:.4f}")
    
    reliability_grade = _compute_reliability_grade(
        performance_spread['mae_gap'],
        consistency_score,
        mae_mean,
        best_model[1]['r2']
    )
    logger.info(f"Reliability grade assigned: {reliability_grade}")
    
    recommendation = _generate_recommendation(
        best_model[0],
        best_model[1],
        performance_spread,
        consistency_score,
        mae_mean
    )
    logger.info("Recommendation generated")
    
    analysis_result = {
        'best_model': best_model[0],
        'worst_model': worst_model[0],
        'ranking': [(name, metrics['mae']) for name, metrics in ranking],
        'performance_spread': performance_spread,
        'consistency_score': consistency_score,
        'statistical_summary': statistical_summary,
        'recommendation': recommendation,
        'reliability_grade': reliability_grade
    }
    
    logger.info("="*60)
    logger.info("MODEL PERFORMANCE ANALYSIS COMPLETE")
    logger.info("="*60)
    
    return analysis_result


def _compute_reliability_grade(mae_gap: float, consistency: float, mean_mae: float, best_r2: float) -> str:
    """
    Compute reliability grade based on performance characteristics.
    
    Parameters
    ----------
    mae_gap : float
        Absolute MAE difference between best and worst models.
    consistency : float
        Normalized consistency score between 0 and 1.
    mean_mae : float
        Mean MAE across all models.
    best_r2 : float
        R2 score of best performing model.
        
    Returns
    -------
    str
        Reliability grade ranging from A+ to F.
        
    Notes
    -----
    - Grade synthesizes consistency, relative gap, and predictive power
    - Higher consistency and R2 with lower gap yield better grades
    - R2 component ensures model has meaningful predictive capability
    - Relative gap provides scale-invariant performance assessment
    """
    relative_gap = mae_gap / mean_mae if mean_mae > 0 else 0.0
    
    r2_bonus = max(0, min(20, best_r2 * 20)) if best_r2 > 0 else 0
    
    composite_score = 100 * consistency - 40 * relative_gap + r2_bonus
    
    if composite_score >= 95:
        return "A+"
    elif composite_score >= 90:
        return "A"
    elif composite_score >= 85:
        return "A-"
    elif composite_score >= 80:
        return "B+"
    elif composite_score >= 75:
        return "B"
    elif composite_score >= 70:
        return "B-"
    elif composite_score >= 65:
        return "C+"
    elif composite_score >= 60:
        return "C"
    elif composite_score >= 55:
        return "C-"
    elif composite_score >= 50:
        return "D"
    else:
        return "F"


def _generate_recommendation(
    best_name: str,
    best_metrics: Dict[str, float],
    spread: Dict[str, float],
    consistency: float,
    mean_mae: float
) -> str:
    """
    Generate deployment recommendation based on comprehensive analysis.
    
    Parameters
    ----------
    best_name : str
        Name of best performing model.
    best_metrics : Dict[str, float]
        Metrics dictionary containing mae and r2 for best model.
    spread : Dict[str, float]
        Performance spread metrics including gaps.
    consistency : float
        Normalized consistency score.
    mean_mae : float
        Mean MAE across all models.
        
    Returns
    -------
    str
        Detailed recommendation for production deployment.
        
    Notes
    -----
    - Considers both absolute performance and model consistency
    - R2 score influences confidence in predictive capability
    - Large spread suggests clear model superiority
    - High consistency with low spread indicates comparable models
    - Negative R2 triggers warnings about model validity
    """
    mae = best_metrics['mae']
    r2 = best_metrics['r2']
    relative_gap = spread['mae_relative_gap']
    
    if r2 < 0:
        return f"{best_name} identified as optimal but exhibits negative R2 ({r2:.4f}), indicating poor predictive capability. Model performance worse than naive mean baseline. Recommend comprehensive feature engineering review, alternative architectures, or ensemble methods before production deployment."
    
    if consistency > 0.95 and relative_gap < 0.05 and r2 > 0.3:
        return f"Exceptional model convergence observed. {best_name} recommended with high confidence (MAE: {mae:.6f}, R2: {r2:.4f}). Minimal performance variance across architectures suggests robust feature set and well-conditioned problem space. All evaluated models demonstrate production viability."
    
    if r2 > 0.5 and relative_gap > 0.25:
        return f"{best_name} demonstrates significant architectural superiority (MAE: {mae:.6f}, R2: {r2:.4f}, relative gap: {relative_gap:.2%}). Strong recommendation for exclusive production deployment. Alternative models exhibit substantial performance degradation and offer no competitive advantage."
    
    if consistency < 0.70:
        return f"Moderate inconsistency detected across model architectures. {best_name} recommended (MAE: {mae:.6f}, R2: {r2:.4f}) but results indicate sensitivity to model selection. Consider ensemble approaches, extended cross-validation, or architectural meta-optimization for robust deployment."
    
    if relative_gap > 0.15 and r2 > 0.2:
        return f"{best_name} clearly outperforms alternatives with notable margin (MAE: {mae:.6f}, R2: {r2:.4f}). Performance characteristics justify preferential deployment. Gap metrics support confident production rollout with continuous monitoring protocols."
    
    if r2 < 0.1:
        return f"{best_name} identified as optimal (MAE: {mae:.6f}) but low R2 ({r2:.4f}) indicates weak predictive power. Model captures minimal variance in target variable. Recommend feature augmentation, regime-specific modeling, or alternative problem formulation before deployment."
    
    return f"{best_name} selected as production candidate with acceptable performance envelope (MAE: {mae:.6f}, R2: {r2:.4f}). Consistency and spread metrics support deployment with standard monitoring. System exhibits normal model comparison characteristics suitable for live trading research."