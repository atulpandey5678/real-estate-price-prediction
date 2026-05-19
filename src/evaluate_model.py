"""
Module for evaluating trained models and selecting the best performer based on R2 Score.
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates MAE, RMSE, and R2 score for given predictions."""
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "R2_Score": round(r2_score(y_true, y_pred), 4),
    }


def evaluate_models(models: Dict[str, Any], X_test: Any, y_test: Any) -> Dict[str, Dict[str, float]]:
    """Evaluates a dictionary of models on the test set and returns their metrics."""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = calculate_metrics(y_test, y_pred)
    return results


def select_best_model(results: Dict[str, Dict[str, float]], models: Dict[str, Any]) -> Tuple[str, Any, Dict[str, float]]:
    """Selects the best model based on the highest R2 Score."""
    best_name = max(results, key=lambda name: results[name]["R2_Score"])
    return best_name, models[best_name], results[best_name]
