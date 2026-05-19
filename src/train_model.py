"""
Module to execute the model training pipeline, evaluate multiple algorithms, 
and persist the best performing model.
"""

import os
import json
import joblib
from typing import Dict, Any, List, Tuple

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.feature_engineering import get_engineered_data
from src.evaluate_model import evaluate_models, select_best_model


def get_models() -> Dict[str, Any]:
    """Returns a dictionary of uninitialized machine learning models to train."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        ),
    }


def train_models(models: Dict[str, Any], X_train: Any, y_train: Any) -> Dict[str, Any]:
    """Trains the given models using the training data."""
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def save_model(model: Any, path: str) -> None:
    """Serializes and saves the model to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def save_feature_columns(columns: List[str], path: str) -> None:
    """Saves the list of expected feature columns to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(columns, f)


def run_training_pipeline(data_path: str, model_dir: str) -> Tuple[Any, Dict[str, Dict[str, float]]]:
    """Main function that drives the training and evaluation process."""
    X_train, X_test, y_train, y_test, feature_columns = get_engineered_data(data_path)

    models = get_models()
    trained_models = train_models(models, X_train, y_train)

    results = evaluate_models(trained_models, X_test, y_test)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    best_name, best_model, best_metrics = select_best_model(results, trained_models)

    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_name}")
    print(f"R² Score: {best_metrics['R2_Score']}")
    print(f"{'=' * 60}")

    model_path = os.path.join(model_dir, "house_price_model.pkl")
    save_model(best_model, model_path)
    print(f"\nModel saved to: {model_path}")

    columns_path = os.path.join(model_dir, "feature_columns.json")
    save_feature_columns(feature_columns, columns_path)
    print(f"Feature columns saved to: {columns_path}")

    return best_model, results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "train.csv")
    model_dir = os.path.join(base_dir, "models")
    run_training_pipeline(data_path, model_dir)
