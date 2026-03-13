import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import get_engineered_data
from src.evaluate_model import evaluate_models, select_best_model
from src.train_model import get_models, train_models

X_train, X_test, y_train, y_test, cols = get_engineered_data("data/train.csv")
models = get_models()
trained = train_models(models, X_train, y_train)
results = evaluate_models(trained, X_test, y_test)

lines = []
lines.append("=" * 60)
lines.append("MODEL EVALUATION RESULTS")
lines.append("=" * 60)
for name, metrics in results.items():
    lines.append(f"\n{name}:")
    for metric, value in metrics.items():
        lines.append(f"  {metric}: {value}")

best_name, _, best_metrics = select_best_model(results, trained)
lines.append(f"\n{'=' * 60}")
lines.append(f"BEST MODEL: {best_name}")
lines.append(f"R2 Score: {best_metrics['R2_Score']}")
lines.append(f"{'=' * 60}")

output = "\n".join(lines)
with open("metrics_output.txt", "w") as f:
    f.write(output)
print("Saved to metrics_output.txt")
