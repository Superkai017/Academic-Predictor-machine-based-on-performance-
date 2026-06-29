"""
prediction.py — Model loading and inference for tier classification.

Loads the tier classifier and scaler once at import time. Exposes:
  - predict_tier(student)    → tier classification result
  - full_prediction(student) → tier + recommendations + feature profile

FRONTEND CONTRACT:
    The frontend calls POST /api/predict with 20 feature values.
    app.py delegates to full_prediction() which returns everything the
    frontend needs in one response: tier prediction, recommendations,
    and feature profile.
"""

import os
import numpy as np
import joblib

from feature_config import FEATURE_ORDER, CLASS_NAMES
from recommendations import generate_recommendations, compute_feature_profile


# ── Load model artifacts at import time ──────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

tier_model = joblib.load(os.path.join(MODELS_DIR, "tier_classifier.pkl"))
tier_scaler = joblib.load(os.path.join(MODELS_DIR, "tier_scaler.pkl"))


def softmax_with_temp(probs: np.ndarray, temperature: float = 2.0) -> np.ndarray:
    """
    Apply temperature scaling to soften a probability distribution.

    Higher temperature → flatter distribution (less overconfident).
    Used to calibrate the classifier's raw probabilities before displaying
    them to the user, since the model can be overly confident on small data.
    """
    logits = np.log(np.clip(probs, 1e-9, 1)) / temperature
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def predict_tier(student: dict) -> dict:
    """
    Predict the student's performance tier (At Risk / Moderate / Excellent).

    Parameters
    ----------
    student : dict
        20 base features keyed by FEATURE_ORDER names.

    Returns
    -------
    dict with keys:
        prediction    : int (0, 1, or 2)
        label         : str ("At Risk", "Moderate", or "Excellent")
        confidence    : float (0-1, temperature-scaled probability of predicted class)
        probabilities : dict mapping class labels to probabilities
    """
    X = np.array([[student[feat] for feat in FEATURE_ORDER]])
    X_scaled = tier_scaler.transform(X)

    pred_idx = int(tier_model.predict(X_scaled)[0])
    proba_raw = tier_model.predict_proba(X_scaled)[0]
    proba = softmax_with_temp(proba_raw)

    labels = tier_model.classes_
    return {
        "prediction": pred_idx,
        "label": CLASS_NAMES[pred_idx],
        "confidence": round(float(proba[list(labels).index(pred_idx)]), 4),
        "probabilities": {
            CLASS_NAMES[int(label)]: round(float(prob), 4)
            for label, prob in zip(labels, proba)
        },
    }


def full_prediction(student: dict) -> dict:
    """
    Run tier prediction and generate recommendations in one call.

    This is the main function called by POST /api/predict. It returns
    everything the frontend needs to render the results page:
      - tier prediction with probabilities
      - per-feature recommendations
      - feature profile (% of valid range)

    Parameters
    ----------
    student : dict
        20 base features from user input.

    Returns
    -------
    dict with keys: status, tier, recommendations, feature_profile
    """
    tier_result = predict_tier(student)
    recs = generate_recommendations(student, tier_result["prediction"])
    profile = compute_feature_profile(student)

    return {
        "status": "success",
        "tier": tier_result,
        "recommendations": recs,
        "feature_profile": profile,
    }
