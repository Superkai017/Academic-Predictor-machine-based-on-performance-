"""
config.py — Central configuration for the Academic Performance Classifier.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\MSI\ML-final-project\Academic-Predictor-machine-based-on-performance-\data\preprocessed\preprocessed_data(1).csv"

TARGET_COL  = "Performance_Tier"
DROP_COLS   = ["GPA", "Performance_Tier", "Available_Study_Time"]
CLASS_NAMES = ["At Risk", "Moderate", "Excellent"]

# ── Feature selection ─────────────────────────────────────────────────────────
K_FEATURES = 20

# ── Train / test split ────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
