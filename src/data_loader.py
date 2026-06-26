"""
data_loader.py — Data loading and feature-preparation pipeline.
"""

import pandas as pd
from functools import partial

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from config import (
    DATA_PATH, TARGET_COL, DROP_COLS,
    K_FEATURES, TEST_SIZE, RANDOM_STATE,
)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the preprocessed CSV from disk."""
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame):
    """
    Full preparation pipeline: split → feature selection → SMOTE → scaling.

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled  : np.ndarray
    y_train_bal    : pd.Series  (after SMOTE)
    y_test         : pd.Series
    selector       : fitted SelectKBest
    scaler         : fitted StandardScaler
    selected_features : pd.Index
    """
    X = df.drop(DROP_COLS, axis=1)
    Y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=Y,
    )

    # Feature selection — fit only on train split
    selector = SelectKBest(
        score_func=partial(mutual_info_classif, random_state=RANDOM_STATE),
        k=K_FEATURES,
    )
    X_sel_train = selector.fit_transform(X_train, y_train)
    X_sel_test  = selector.transform(X_test)

    selected_features = X_train.columns[selector.get_support(indices=True)]
    print("Selected features:", list(selected_features))

    # SMOTE — applied to train only to avoid data leakage
    smote = SMOTE(random_state=RANDOM_STATE)
    X_bal, y_bal = smote.fit_resample(X_sel_train, y_train)          # type: ignore
    print("\nAfter SMOTE:", pd.Series(y_bal).value_counts().to_dict())  # type: ignore

    # Scaling — fit on balanced train, transform both splits
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_bal)                       # type: ignore
    X_test_scaled  = scaler.transform(X_sel_test)

    return (
        X_train_scaled, X_test_scaled,
        y_bal, y_test,
        selector, scaler, selected_features,
    )
