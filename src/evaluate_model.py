import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "R2_Score": round(r2_score(y_true, y_pred), 4),
    }


def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = calculate_metrics(y_test, y_pred)
    return results


def select_best_model(results, models):
    best_name = max(results, key=lambda name: results[name]["R2_Score"])
    return best_name, models[best_name], results[best_name]
