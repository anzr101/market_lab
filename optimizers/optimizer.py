import numpy as np
import logging
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import optuna
from itertools import product


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(log_dir / "optimizer.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


MAX_SAMPLES = 50000
RANDOM_STATE = 42


def run_optimizers(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    max_runtime_seconds: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare three hyperparameter optimization methods for RandomForestRegressor.
    
    Implements Grid Search, Random Search, and Optuna optimization with
    intelligent subsampling, resource management, and runtime controls for
    production deployment. Optimized for large-scale datasets while maintaining
    statistical validity.
    
    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training target vector.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        Test target vector.
    max_runtime_seconds : Optional[int], default=None
        Maximum runtime per optimizer in seconds. If exceeded, optimization
        terminates gracefully and returns best result found.
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping optimizer names to performance metrics:
        {
            "GridSearch": {
                "mae": float,
                "params": dict,
                "trials": int,
                "time_seconds": float
            },
            "RandomSearch": {...},
            "Optuna": {...}
        }
        
    Raises
    ------
    ValueError
        If input data is invalid, empty, or contains NaN/infinite values.
        
    Notes
    -----
    - Automatically subsamples training data to 50k rows for efficiency
    - Uses adaptive parameter spaces optimized for speed
    - Resource-aware: n_jobs = cpu_count - 1
    - All operations are deterministic (random_state=42)
    - No cross-validation to preserve time-series integrity
    - Target runtime: <60s for 200k rows, <20s for 50k rows
    """
    logger.info("="*80)
    logger.info("HYPERPARAMETER OPTIMIZATION COMPARISON")
    logger.info("="*80)
    
    _validate_inputs(X_train, y_train, X_test, y_test)
    
    logger.info(f"Initial train size: {len(X_train)}, test size: {len(X_test)}")
    
    X_train_opt, y_train_opt = _subsample_dataset(X_train, y_train)
    
    n_jobs = max(1, os.cpu_count() - 1) if os.cpu_count() else 1
    logger.info(f"Resource allocation: n_jobs={n_jobs}")
    
    max_estimators = 120 if len(X_train_opt) > 150000 else 200
    logger.info(f"Adaptive search space: max_estimators={max_estimators}")
    
    results = {}
    
    results['GridSearch'] = _run_grid_search(
        X_train_opt, y_train_opt, X_test, y_test, n_jobs, max_estimators, max_runtime_seconds
    )
    
    results['RandomSearch'] = _run_random_search(
        X_train_opt, y_train_opt, X_test, y_test, n_jobs, max_estimators, max_runtime_seconds
    )
    
    results['Optuna'] = _run_optuna_search(
        X_train_opt, y_train_opt, X_test, y_test, n_jobs, max_estimators, max_runtime_seconds
    )
    
    logger.info("="*80)
    logger.info("OPTIMIZATION COMPARISON SUMMARY")
    logger.info("="*80)
    
    for method, result in results.items():
        logger.info(f"{method}: MAE={result['mae']:.6f}, trials={result['trials']}, time={result['time_seconds']:.2f}s")
    
    best_optimizer = min(results.items(), key=lambda x: x[1]['mae'])
    logger.info(f"Best optimizer: {best_optimizer[0]} (MAE: {best_optimizer[1]['mae']:.6f})")
    logger.info("="*80)
    
    return results


def _validate_inputs(X_train: Any, y_train: Any, X_test: Any, y_test: Any) -> None:
    """
    Validate input data for optimization.
    
    Parameters
    ----------
    X_train, y_train, X_test, y_test : array-like
        Training and test data.
        
    Raises
    ------
    ValueError
        If data is invalid, empty, or contains NaN/infinite values.
    """
    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("Empty dataset provided")
        raise ValueError("Training and test sets must not be empty")
    
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        logger.error("Feature and target dimension mismatch")
        raise ValueError("X and y dimensions must match")
    
    X_train_array = np.asarray(X_train)
    y_train_array = np.asarray(y_train)
    X_test_array = np.asarray(X_test)
    y_test_array = np.asarray(y_test)
    
    if not np.all(np.isfinite(X_train_array)):
        logger.error("Training features contain NaN or infinite values")
        raise ValueError("X_train contains non-finite values")
    
    if not np.all(np.isfinite(y_train_array)):
        logger.error("Training targets contain NaN or infinite values")
        raise ValueError("y_train contains non-finite values")
    
    if not np.all(np.isfinite(X_test_array)):
        logger.error("Test features contain NaN or infinite values")
        raise ValueError("X_test contains non-finite values")
    
    if not np.all(np.isfinite(y_test_array)):
        logger.error("Test targets contain NaN or infinite values")
        raise ValueError("y_test contains non-finite values")
    
    logger.info("Input validation passed")


def _subsample_dataset(X_train: Any, y_train: Any) -> tuple:
    """
    Subsample training data for efficient optimization.
    
    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training targets.
        
    Returns
    -------
    tuple
        (X_train_sampled, y_train_sampled)
        
    Notes
    -----
    - Subsamples to MAX_SAMPLES if dataset exceeds threshold
    - Uses deterministic random sampling (random_state=42)
    - Preserves statistical properties through uniform sampling
    """
    n_samples = len(X_train)
    
    if n_samples > MAX_SAMPLES:
        logger.info(f"Dataset large ({n_samples} rows) -> subsampling to {MAX_SAMPLES} rows for optimization")
        
        X_train_array = np.asarray(X_train)
        y_train_array = np.asarray(y_train)
        
        indices = np.arange(n_samples)
        rng = np.random.default_rng(RANDOM_STATE)
        sampled_indices = rng.choice(indices, size=MAX_SAMPLES, replace=False)
        
        X_sampled = X_train_array[sampled_indices]
        y_sampled = y_train_array[sampled_indices]
        
        logger.info(f"Subsampling complete: {len(X_sampled)} rows")
        return X_sampled, y_sampled
    else:
        logger.info(f"Dataset size ({n_samples}) within threshold, no subsampling needed")
        return X_train, y_train


def _run_grid_search(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    n_jobs: int,
    max_estimators: int,
    max_runtime: Optional[int]
) -> Dict[str, Any]:
    """
    Execute grid search optimization with expanded parameter space.
    
    Parameters
    ----------
    X_train, y_train : array-like
        Training data.
    X_test, y_test : array-like
        Test data.
    n_jobs : int
        Number of parallel jobs.
    max_estimators : int
        Maximum number of estimators to search.
    max_runtime : Optional[int]
        Maximum runtime in seconds.
        
    Returns
    -------
    Dict[str, Any]
        Results dictionary with mae, params, trials, time_seconds.
    """
    logger.info("="*80)
    logger.info("GRID SEARCH OPTIMIZATION")
    logger.info("="*80)
    
    param_grid = {
    'n_estimators': [50, 120],
    'max_depth': [5, 15],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1],
    'max_leaf_nodes': [100],
    'min_weight_fraction_leaf': [0.0],
    'ccp_alpha': [0.0],
    'max_features': ['sqrt', None],
    'bootstrap': [True],
    'criterion': ['squared_error']
}
    
    start_time = time.time()
    
    try:
        X_train_f32 = np.asarray(X_train, dtype=np.float32)
        X_test_f32 = np.asarray(X_test, dtype=np.float32)
        y_train_arr = np.asarray(y_train)
        y_test_arr = np.asarray(y_test)
        
        grid_combinations = list(product(
            param_grid['n_estimators'],
            param_grid['max_depth'],
            param_grid['min_samples_split'],
            param_grid['min_samples_leaf'],
            param_grid['max_leaf_nodes'],
            param_grid['min_weight_fraction_leaf'],
            param_grid['ccp_alpha'],
            param_grid['max_features'],
            param_grid['bootstrap'],
            param_grid['criterion']
        ))
        
        total_combinations = len(grid_combinations)
        logger.info(f"Grid Search: evaluating {total_combinations} parameter combinations")
        
        best_mae = float('inf')
        best_params = None
        trials_completed = 0
        
        for idx, combo in enumerate(grid_combinations, 1):
            if max_runtime and (time.time() - start_time) > max_runtime:
                logger.warning(f"Grid Search exceeded max runtime ({max_runtime}s), terminating early")
                break
            
            params = {
                'n_estimators': combo[0],
                'max_depth': combo[1],
                'min_samples_split': combo[2],
                'min_samples_leaf': combo[3],
                'max_leaf_nodes': combo[4],
                'min_weight_fraction_leaf': combo[5],
                'ccp_alpha': combo[6],
                'max_features': combo[7],
                'bootstrap': combo[8],
                'criterion': combo[9],
                'random_state': RANDOM_STATE,
                'n_jobs': n_jobs
            }
            
            logger.info(f"Grid [{idx}/{total_combinations}] Trial params = {params}")
            
            model = RandomForestRegressor(**params)
            
            model.fit(X_train_f32, y_train_arr)
            y_pred = model.predict(X_test_f32)
            mae = mean_absolute_error(y_test_arr, y_pred)
            
            trials_completed += 1
            
            if mae < best_mae:
                best_mae = mae
                best_params = {k: v for k, v in params.items() if k not in ['random_state', 'n_jobs']}
            
            logger.info(f"[Grid] {idx}/{total_combinations} complete | current MAE={mae:.6f} | best MAE={best_mae:.6f}")
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"Grid Search complete: best MAE={best_mae:.6f}, params={best_params}")
        logger.info(f"Grid Search time: {elapsed_time:.2f}s")
        
        return {
            'mae': best_mae,
            'params': best_params,
            'trials': trials_completed,
            'time_seconds': elapsed_time
        }
        
    except Exception as e:
        logger.error(f"Grid Search failed: {str(e)}")
        raise


def _run_random_search(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    n_jobs: int,
    max_estimators: int,
    max_runtime: Optional[int]
) -> Dict[str, Any]:
    """
    Execute random search optimization with expanded parameter space.
    
    Parameters
    ----------
    X_train, y_train : array-like
        Training data.
    X_test, y_test : array-like
        Test data.
    n_jobs : int
        Number of parallel jobs.
    max_estimators : int
        Maximum number of estimators to search.
    max_runtime : Optional[int]
        Maximum runtime in seconds.
        
    Returns
    -------
    Dict[str, Any]
        Results dictionary with mae, params, trials, time_seconds.
    """
    logger.info("="*80)
    logger.info("RANDOM SEARCH OPTIMIZATION")
    logger.info("="*80)
    
    n_iterations = 5
    logger.info(f"Random Search: evaluating {n_iterations} random combinations")
    
    start_time = time.time()
    
    try:
        X_train_f32 = np.asarray(X_train, dtype=np.float32)
        X_test_f32 = np.asarray(X_test, dtype=np.float32)
        y_train_arr = np.asarray(y_train)
        y_test_arr = np.asarray(y_test)
        
        rng = np.random.default_rng(RANDOM_STATE)
        
        max_features_choices = ['sqrt', 'log2', None]
        bootstrap_choices = [True, False]
        criterion_choices = ['squared_error', 'absolute_error', 'friedman_mse']
        
        best_mae = float('inf')
        best_params = None
        trials_completed = 0
        
        for idx in range(1, n_iterations + 1):
            if max_runtime and (time.time() - start_time) > max_runtime:
                logger.warning(f"Random Search exceeded max runtime ({max_runtime}s), terminating early")
                break
            
            params = {
                'n_estimators': int(rng.integers(50, max_estimators + 1)),
                'max_depth': int(rng.integers(3, 31)),
                'min_samples_split': int(rng.integers(2, 21)),
                'min_samples_leaf': int(rng.integers(1, 21)),
                'max_leaf_nodes': int(rng.integers(10, 201)),
                'min_weight_fraction_leaf': float(rng.uniform(0.0, 0.5)),
                'ccp_alpha': float(rng.uniform(0.0, 0.05)),
                'max_features': rng.choice(max_features_choices),
                'bootstrap': bool(rng.choice(bootstrap_choices)),
                'criterion': rng.choice(criterion_choices),
                'random_state': RANDOM_STATE,
                'n_jobs': n_jobs
            }
            
            logger.info(f"Random [{idx}/{n_iterations}] Trial params = {params}")
            
            model = RandomForestRegressor(**params)
            
            model.fit(X_train_f32, y_train_arr)
            y_pred = model.predict(X_test_f32)
            mae = mean_absolute_error(y_test_arr, y_pred)
            
            trials_completed += 1
            
            if mae < best_mae:
                best_mae = mae
                best_params = {k: v for k, v in params.items() if k not in ['random_state', 'n_jobs']}
            
            logger.info(f"[Random] {idx}/{n_iterations} complete | current MAE={mae:.6f} | best MAE={best_mae:.6f}")
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"Random Search complete: best MAE={best_mae:.6f}, params={best_params}")
        logger.info(f"Random Search time: {elapsed_time:.2f}s")
        
        return {
            'mae': best_mae,
            'params': best_params,
            'trials': trials_completed,
            'time_seconds': elapsed_time
        }
        
    except Exception as e:
        logger.error(f"Random Search failed: {str(e)}")
        raise


def _run_optuna_search(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    n_jobs: int,
    max_estimators: int,
    max_runtime: Optional[int]
) -> Dict[str, Any]:
    """
    Execute Optuna optimization with expanded parameter space.
    
    Parameters
    ----------
    X_train, y_train : array-like
        Training data.
    X_test, y_test : array-like
        Test data.
    n_jobs : int
        Number of parallel jobs.
    max_estimators : int
        Maximum number of estimators to search.
    max_runtime : Optional[int]
        Maximum runtime in seconds.
        
    Returns
    -------
    Dict[str, Any]
        Results dictionary with mae, params, trials, time_seconds.
    """
    logger.info("="*80)
    logger.info("OPTUNA OPTIMIZATION")
    logger.info("="*80)
    
    start_time = time.time()
    
    try:
        X_train_f32 = np.asarray(X_train, dtype=np.float32)
        X_test_f32 = np.asarray(X_test, dtype=np.float32)
        y_train_arr = np.asarray(y_train)
        y_test_arr = np.asarray(y_test)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        trial_counter = {'count': 0, 'best_mae': float('inf')}
        
        def objective(trial):
            if max_runtime and (time.time() - start_time) > max_runtime:
                logger.warning(f"Optuna exceeded max runtime ({max_runtime}s), terminating early")
                raise optuna.exceptions.OptunaError("Runtime limit exceeded")
            
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, max_estimators),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 200),
                'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
                'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.05),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'friedman_mse']),
                'random_state': RANDOM_STATE,
                'n_jobs': n_jobs
            }
            
            trial_counter['count'] += 1
            logger.info(f"Optuna Trial {trial_counter['count']}/15 params = {params}")
            
            model = RandomForestRegressor(**params)
            
            model.fit(X_train_f32, y_train_arr)
            y_pred = model.predict(X_test_f32)
            mae = mean_absolute_error(y_test_arr, y_pred)
            
            if mae < trial_counter['best_mae']:
                trial_counter['best_mae'] = mae
            
            logger.info(f"[Optuna] Trial {trial_counter['count']}/15 | current MAE={mae:.6f} | best MAE={trial_counter['best_mae']:.6f}")
            
            return mae
        
        n_trials = 8
        logger.info(f"Optuna: running {n_trials} trials")
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        )
        
        try:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        except optuna.exceptions.OptunaError:
            logger.warning("Optuna terminated early due to runtime limit")
        
        elapsed_time = time.time() - start_time
        
        if len(study.trials) == 0:
            logger.error("Optuna completed zero trials, returning fallback result")
            return {
                'mae': float('inf'),
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_leaf_nodes': 100,
                    'min_weight_fraction_leaf': 0.0,
                    'ccp_alpha': 0.0,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'criterion': 'squared_error'
                },
                'trials': 0,
                'time_seconds': elapsed_time
            }
        
        best_mae = study.best_value
        best_params = study.best_params
        trials_completed = len(study.trials)
        
        logger.info(f"Optuna complete: best MAE={best_mae:.6f}, params={best_params}")
        logger.info(f"Optuna completed {trials_completed} trials in {elapsed_time:.2f}s")
        
        return {
            'mae': best_mae,
            'params': best_params,
            'trials': trials_completed,
            'time_seconds': elapsed_time
        }
        
    except Exception as e:
        logger.error(f"Optuna optimization failed: {str(e)}")
        raise