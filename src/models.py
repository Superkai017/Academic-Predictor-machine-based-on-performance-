"""
models.py — Baseline model definitions.
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from config import RANDOM_STATE


def build_models() -> dict:
    """
    Return a dict of lightly-regularised baseline estimators.

    Keys: 'lr', 'rf', 'gb', 'xgb'
    """
    return {
        "lr": LogisticRegression(
            C=0.1,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "rf": RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=RANDOM_STATE,
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=50,
            max_depth=1,
            learning_rate=0.05,
            subsample=0.6,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
        "xgb": XGBClassifier(
            n_estimators=50,
            max_depth=1,
            learning_rate=0.05,
            subsample=0.6,
            colsample_bytree=0.7,
            reg_alpha=5.0,
            reg_lambda=10.0,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
        ),
    }
