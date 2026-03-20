import numpy as np
from typing import Dict


def sharpe_ratio(returns: np.ndarray) -> float:
    mean = np.mean(returns)
    std = np.std(returns)
    if std == 0:
        return 0.0
    return float((mean / std) * np.sqrt(252))


def max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return float(drawdown.min())


def win_rate(returns: np.ndarray) -> float:
    return float((returns > 0).sum() / len(returns))


def profit_factor(returns: np.ndarray) -> float:
    gains = returns[returns > 0].sum()
    losses = np.abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf")
    return float(gains / losses)


def summary_stats(returns: np.ndarray) -> Dict[str, float]:
    equity = np.cumprod(1 + returns)

    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
    }
