"""
main.py — Entry point: runs the full training pipeline.
"""

from config import DATA_PATH
from data_loader import load_data, prepare_features
from models import build_models
from evaluation import train_and_evaluate, plot_evaluation
from tuning import grid_search


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

    return best_lr, scaler, selector


if __name__ == "__main__":
    train_pipeline()
