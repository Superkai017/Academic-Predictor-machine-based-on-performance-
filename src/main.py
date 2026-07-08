"""
main.py — Entry point: runs the full training pipeline.

After training, the best classification model and its scaler are saved to
src/backend/models/ so the Flask backend can load them for inference.
"""

import os
import joblib

from config import DATA_PATH
from data_loader import load_data, prepare_features
from models import build_models
from evaluation import train_and_evaluate, plot_evaluation
from tuning import grid_search


# Where the Flask backend expects model artifacts
BACKEND_MODELS_DIR = os.path.join(os.path.dirname(__file__))


def train_pipeline(data_path: str = DATA_PATH):
    # 1. Load & prepare
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, selector, scaler, sel_feats = \
        prepare_features(df)

    # 2. Baseline evaluation
    models = build_models()
    train_and_evaluate(models, X_train, X_test, y_train, y_test)

    # 3. Hyperparameter tuning
    best_models = grid_search(X_train, y_train)

    # 4. Evaluate best Logistic Regression
    best_lr = best_models["lr"]
    plot_evaluation(best_lr, X_test, y_test)

    # 5. Export to backend/models/ for the Flask API
    os.makedirs(BACKEND_MODELS_DIR, exist_ok=True)
    joblib.dump(best_lr, os.path.join(BACKEND_MODELS_DIR, "tier_classifier.pkl"))
    joblib.dump(scaler,  os.path.join(BACKEND_MODELS_DIR, "tier_scaler.pkl"))
    print(f"\nModels saved to {BACKEND_MODELS_DIR}/")
    print(f"  tier_classifier.pkl  — best Logistic Regression")
    print(f"  tier_scaler.pkl      — fitted StandardScaler")

    return best_lr, scaler, selector


if __name__ == "__main__":
    train_pipeline()
