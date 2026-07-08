"""
feature_engineering.py — Compute derived features for the GPA regression model.

The regression model was trained on 10 features: 6 base features that come
directly from user input, and 4 engineered features computed from combinations
of base features. This module handles that computation.

The frontend sends only the 20 base features. The backend calls
build_regression_features() to produce the 10-feature dict the GPA model needs.
"""

from feature_config import GPA_REGRESSION_FEATURES


def build_regression_features(student: dict) -> dict:
    """
    Compute the 10 features required by the GPA regression model.

    Takes the raw 20-feature student input dict and returns a dict with
    exactly the 10 features in GPA_REGRESSION_FEATURES order:
      - 6 base features (passed through directly)
      - 4 engineered features (computed from base features)

    Parameters
    ----------
    student : dict
        The 20 base features from user input (same dict sent to /api/predict).

    Returns
    -------
    dict
        10-feature dict keyed by GPA_REGRESSION_FEATURES names.
    """
    return {
        # ── Base features (passed through) ───────────────────────────────
        "Active_Engagement_Score": student["Active_Engagement_Score"],
        "Study_Quality":          student["Study_Quality"],
        "Resource_Quality":       student["Resource_Quality"],
        "Lifestyle_Index":        student["Lifestyle_Index"],
        "Academic_Momentum":      student["Academic_Momentum"],
        "Stress_Load":            student["Stress_Load"],

        # ── Engineered features ──────────────────────────────────────────
        # Study quality sustained over time
        "Quality_Consistency": student["Study_Quality"] * student["Study_Consistency"],

        # Effective use of AI tools
        "AI_Leverage": student["AI_Enhanced_Study"] * student["AI_Engagement"],

        # Combined resource access, AI enhancement, and consistency
        "Learning_Power": (
            student["Resource_Quality"]
            * student["AI_Enhanced_Study"]
            * student["Study_Consistency"]
        ),

        # Stress amplified by poor lifestyle habits
        "Stress_Penalty": student["Stress_Load"] * (1 - student["Lifestyle_Index"]),
    }
