"""
tuning.py — Hyperparameter search for all four classifiers.
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from config import RANDOM_STATE


# ── Parameter grids ───────────────────────────────────────────────────────────

RF_PARAMS = {
    "n_estimators":    [100, 200],
    "max_depth":       [3, 4, 5],
    "min_samples_split": [10, 20],
    "min_samples_leaf":  [5, 10],
    "max_features":    ["sqrt", "log2"],
}

LR_PARAMS = {
    "C":        [0.01, 0.1, 1, 10],
    "solver":   ["lbfgs", "saga"],
    "max_iter": [500, 1000],
}

GB_PARAMS = {
    "n_estimators":    [100, 200],
    "max_depth":       [2, 3],
    "learning_rate":   [0.01, 0.05],
    "subsample":       [0.7, 0.8],
    "min_samples_leaf": [5, 10],
}

XGB_PARAMS = {
    "n_estimators":  [100, 200],
    "max_depth":     [2, 3],
    "learning_rate": [0.01, 0.05],
    "subsample":     [0.7, 0.8],
    "reg_alpha":     [0.1, 1.0],
    "reg_lambda":    [1.0, 5.0],
}


# ── Grid search runner ────────────────────────────────────────────────────────

def grid_search(X_train, y_train) -> dict:
    """
    Run 5-fold GridSearchCV for each classifier.

    Returns
    -------
    dict mapping model key → best fitted estimator
    """
    grids = {
        "rf": GridSearchCV(
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            RF_PARAMS, cv=5, scoring="accuracy", n_jobs=-1,
        ),
        "lr": GridSearchCV(
            LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE),
            LR_PARAMS, cv=5,
        ),
        "gb": GridSearchCV(
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            GB_PARAMS, cv=5,
        ),
        "xgb": GridSearchCV(
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss"),
            XGB_PARAMS, cv=5,
        ),
    }

    best: dict = {}
    for name, grid in grids.items():
        print(f"Grid searching {name} …")
        grid.fit(X_train, y_train)
        print(f"  Best {name}: {grid.best_params_}  score={grid.best_score_:.4f}")
        best[name] = grid.best_estimator_

    return best
