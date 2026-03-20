import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "optimizer_analysis.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def analyze_optimizer_results(results: Dict[str, float]) -> Dict[str, Any]:
    """
    Analyze optimizer comparison results and generate insights.
    
    Computes ranking, performance metrics, stability scores, and provides
    recommendations based on optimizer performance. Lower MAE values indicate
    better performance.
    
    Parameters
    ----------
    results : Dict[str, float]
        Dictionary mapping optimizer names to their MAE scores.
        Lower values indicate better performance.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - best_optimizer: str, name of best performing optimizer
        - worst_optimizer: str, name of worst performing optimizer
        - ranking: List[Tuple[str, float]], sorted list of (name, mae)
        - performance_gap: float, absolute difference between best and worst
        - normalized_stability_score: float, consistency measure (0-1, higher is better)
        - statistical_spread: Dict[str, float], contains mean, std, cv, range
        - recommendation: str, detailed recommendation text
        - reliability_grade: str, grade from A+ to F
        
    Raises
    ------
    TypeError
        If results is not a dictionary or contains invalid types.
    ValueError
        If results is empty or contains invalid numeric values.
        
    Notes
    -----
    - Stability score of 1.0 indicates perfect consistency across optimizers
    - Coefficient of variation used for stability assessment
    - Reliability grade incorporates both performance gap and stability
    - All MAE values must be non-negative and finite
    - Performance gap is relative to mean MAE for scale invariance
    """
    logger.info("="*60)
    logger.info("OPTIMIZER ANALYSIS INITIATED")
    logger.info("="*60)
    
    if not isinstance(results, dict):
        logger.error("Invalid input type: results must be dictionary")
        raise TypeError("results must be a dictionary mapping optimizer names to MAE values")
    
    if len(results) == 0:
        logger.error("Empty results dictionary provided")
        raise ValueError("results dictionary cannot be empty")
    
    logger.info(f"Analyzing {len(results)} optimizer results")
    
    for name, value in results.items():
        mae = value["mae"] if isinstance(value, dict) else value
        if not isinstance(name, str):
            logger.error(f"Invalid optimizer name type: {type(name)}")
            raise TypeError(f"Optimizer name must be string, received {type(name)}")
        
        if not isinstance(mae, (int, float, np.floating)):
            logger.error(f"Invalid MAE type for {name}: {type(mae)}")
            raise ValueError(f"MAE value for '{name}' must be numeric, received {type(mae)}")
        
        if not np.isfinite(mae):
            logger.error(f"Non-finite MAE value for {name}: {mae}")
            raise ValueError(f"MAE value for '{name}' must be finite")
        
        if mae < 0:
            logger.error(f"Negative MAE value for {name}: {mae}")
            raise ValueError(f"MAE value for '{name}' cannot be negative")
        
        logger.info(f"  {name}: MAE = {mae:.6f}")
    
    best_optimizer = min(
    results.items(),
    key=lambda x: x[1]["mae"] if isinstance(x[1], dict) else x[1]
)
    worst_optimizer = max(
    results.items(),
    key=lambda x: x[1]["mae"] if isinstance(x[1], dict) else x[1]
)
    
    logger.info(f"Best performer: {best_optimizer[0]} (MAE: {best_optimizer[1]:.6f})")
    logger.info(f"Worst performer: {worst_optimizer[0]} (MAE: {worst_optimizer[1]:.6f})")
    
    ranking = sorted(
    results.items(),
    key=lambda x: x[1]["mae"] if isinstance(x[1], dict) else x[1]
)
    logger.info(f"Ranking order: {[name for name, _ in ranking]}")
    
    performance_gap = worst_optimizer[1] - best_optimizer[1]
    logger.info(f"Absolute performance gap: {performance_gap:.6f}")
    
    mae_values = np.array([
    v["mae"] if isinstance(v, dict) else v
    for v in results.values()
])
    mean_mae = float(np.mean(mae_values))
    std_mae = float(np.std(mae_values, ddof=1 if len(mae_values) > 1 else 0))
    cv = std_mae / mean_mae if mean_mae > 0 else 0.0
    range_mae = float(np.ptp(mae_values))
    
    statistical_spread = {
        'mean': mean_mae,
        'std': std_mae,
        'cv': cv,
        'range': range_mae
    }
    
    logger.info(f"Statistical spread: mean={mean_mae:.6f}, std={std_mae:.6f}, cv={cv:.4f}, range={range_mae:.6f}")
    
    if len(results) == 1:
        normalized_stability_score = 1.0
        logger.info("Single optimizer detected, stability score set to 1.0")
    else:
        normalized_stability_score = 1.0 / (1.0 + cv)
        logger.info(f"Normalized stability score: {normalized_stability_score:.4f}")
    
    reliability_grade = _compute_reliability_grade(performance_gap, normalized_stability_score, mean_mae)
    logger.info(f"Reliability grade assigned: {reliability_grade}")
    
    recommendation = _generate_recommendation(
        best_optimizer[0],
        performance_gap,
        normalized_stability_score,
        mean_mae
    )
    logger.info("Recommendation generated")
    
    analysis_result = {
        'best_optimizer': best_optimizer[0],
        'worst_optimizer': worst_optimizer[0],
        'ranking': ranking,
        'performance_gap': performance_gap,
        'normalized_stability_score': normalized_stability_score,
        'statistical_spread': statistical_spread,
        'recommendation': recommendation,
        'reliability_grade': reliability_grade
    }
    
    logger.info("="*60)
    logger.info("OPTIMIZER ANALYSIS COMPLETE")
    logger.info("="*60)
    
    return analysis_result


def _compute_reliability_grade(gap: float, stability: float, mean: float) -> str:
    """
    Compute reliability grade based on performance characteristics.
    
    Parameters
    ----------
    gap : float
        Absolute performance gap between best and worst optimizers.
    stability : float
        Normalized stability score between 0 and 1.
    mean : float
        Mean MAE across all optimizers.
        
    Returns
    -------
    str
        Reliability grade ranging from A+ to F.
        
    Notes
    -----
    - Grade based on weighted combination of stability and relative gap
    - Higher stability and lower relative gap yield better grades
    - Relative gap computed as gap/mean for scale invariance
    """
    relative_gap = gap / mean if mean > 0 else 0.0
    
    composite_score = 100 * stability - 50 * relative_gap
    
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


def _generate_recommendation(best_name: str, gap: float, stability: float, mean: float) -> str:
    """
    Generate contextualized recommendation based on analysis.
    
    Parameters
    ----------
    best_name : str
        Name of best performing optimizer.
    gap : float
        Absolute performance gap.
    stability : float
        Normalized stability score.
    mean : float
        Mean MAE across optimizers.
        
    Returns
    -------
    str
        Detailed recommendation text for deployment.
        
    Notes
    -----
    - Recommendation considers both absolute and relative performance
    - High stability with low gap suggests all methods viable
    - Large gap with high stability indicates clear winner
    - Low stability suggests hyperparameter sensitivity
    """
    relative_gap = gap / mean if mean > 0 else 0.0
    
    if stability > 0.95 and relative_gap < 0.05:
        return f"Exceptional consistency detected across all optimizers. {best_name} recommended with high confidence, though all methods yield comparable results. This indicates a robust and well-behaved hyperparameter landscape with minimal optimization strategy sensitivity."
    
    if stability > 0.90 and relative_gap < 0.10:
        return f"{best_name} is the optimal choice with strong confidence. Performance variance is minimal across optimization strategies, suggesting stable search landscape. Alternative optimizers remain acceptable for production use."
    
    if relative_gap > 0.25:
        return f"{best_name} demonstrates significant superiority over alternatives (relative gap: {relative_gap:.2%}). Strong recommendation for exclusive deployment. Competing methods exhibit substantial performance degradation and should be avoided."
    
    if stability < 0.70:
        return f"Moderate inconsistency observed across optimization strategies. {best_name} recommended but results indicate sensitivity to search methodology. Consider extended hyperparameter exploration, ensemble approaches, or alternative model architectures."
    
    if relative_gap > 0.15:
        return f"{best_name} clearly outperforms alternatives with notable margin. Recommended for production deployment. Performance gap justifies preferential use over competing optimizers."
    
    return f"{best_name} identified as optimal optimizer with acceptable performance characteristics. Both gap metrics and stability scores support confident production deployment. System exhibits standard optimization behavior."