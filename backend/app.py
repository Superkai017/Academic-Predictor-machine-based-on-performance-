"""
app.py — Flask backend for Academic Performance Predictor.

Provides REST endpoints for the frontend to:
  1. Submit student features and get predictions (tier + GPA + feedback)
  2. Fetch feature metadata for dynamic form generation
  3. Fetch sample student profiles for demo/testing

Run with:
    cd src/backend
    python app.py

Or from project root:
    .venv/Scripts/python src/backend/app.py

The server starts on http://localhost:5000 by default.

FRONTEND INTEGRATION GUIDE:
    ─────────────────────────────────────────────────────────────────
    POST /api/predict
        Send: JSON body with 20 feature values (see GET /api/features for names)
        Receive: tier prediction, GPA prediction, recommendations, feature profile
        Show a loading spinner while waiting — inference takes <1s.

    GET /api/features
        Receive: feature metadata (name, min, max, step, unit, description)
        Use this to dynamically build the input form with sliders/inputs.

    GET /api/classes
        Receive: the 3 class labels the model can predict.

    GET /api/sample-students
        Receive: 3 pre-filled student profiles (excellent, moderate, at-risk)
        Use these to populate the form for demo purposes.

    GET /api/health
        Receive: { "status": "ok" } — use for connectivity checks.
    ─────────────────────────────────────────────────────────────────

    PERSISTENCE:
        The backend is stateless — it does NOT store prediction results.
        The frontend should use IndexedDB to save results locally so they
        persist across browser tab closes. Each /api/predict response
        contains all data needed for a saved result entry.
"""

import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from feature_config import (
    FEATURE_ORDER,
    FEATURE_METADATA,
    FEATURE_RANGES,
    CLASS_NAMES,
    SAMPLE_STUDENTS,
)
from prediction import full_prediction


# ── Environment config ───────────────────────────────────────────────────────
# Loads backend/.env — see .env.example for available settings.
# On hosting platforms (Render, etc.) set these as environment variables instead.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

PORT = int(os.getenv("PORT", "5000"))
DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
# "*" allows all origins (dev only); otherwise a comma-separated list of URLs.
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")


# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)

if CORS_ORIGINS == "*":
    CORS(app)
else:
    CORS(app, origins=[o.strip() for o in CORS_ORIGINS.split(",")])


# ── Input validation ─────────────────────────────────────────────────────────
def validate_features(data: dict) -> tuple:
    """
    Validate that the request body contains all 20 features within valid ranges.

    Returns
    -------
    (student_dict, None) on success
    (None, error_response_dict) on failure — caller should return this as 400
    """
    if not data:
        return None, {"status": "error", "message": "Request body is empty or not valid JSON."}

    errors = []
    student = {}

    for feat in FEATURE_ORDER:
        val = data.get(feat)

        if val is None:
            errors.append(f"Missing required feature: '{feat}'")
            continue

        # Coerce to float
        try:
            val = float(val)
        except (ValueError, TypeError):
            errors.append(f"'{feat}' must be a number, got: {val!r}")
            continue

        # Range check
        lo, hi = FEATURE_RANGES[feat]
        if val < lo or val > hi:
            errors.append(f"'{feat}' value {val} is out of range [{lo}, {hi}]")
            continue

        student[feat] = val

    if errors:
        return None, {
            "status": "error",
            "message": "Input validation failed.",
            "errors": errors,
        }

    return student, None


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    """
    Health check endpoint.

    Frontend can ping this on page load to confirm the backend is reachable.

    Response: { "status": "ok" }
    """
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint — accepts 20 features, returns full results.

    ── What the frontend should send ──
    Content-Type: application/json
    Body: {
        "Diet_Quality": 3.0,
        "Stress_Level": 5.0,
        "AI_Tool_Usage": 6.0,
        ... (all 20 features)
    }

    ── What the frontend receives on success (200) ──
    {
        "status": "success",
        "tier": {
            "prediction": 2,
            "label": "Excellent",
            "confidence": 0.85,
            "probabilities": {
                "At Risk": 0.05,
                "Moderate": 0.10,
                "Excellent": 0.85
            }
        },
        "recommendations": [
            {
                "feature": "Stress_Level",
                "display_name": "Stress Level",
                "value": 5.0,
                "unit": "/10",
                "severity": "good",        // "warning" | "improve" | "good" | "watch"
                "message": "Stress is well-managed..."
            },
            ...
        ],
        "feature_profile": [
            {
                "feature": "Diet_Quality",
                "value": 3.0,
                "percentage": 66.7,     // % of valid range (for bar visualization)
                "min": 1.0,
                "max": 4.0
            },
            ...
        ]
    }

    ── Error responses ──
    400: { "status": "error", "message": "...", "errors": [...] }
    500: { "status": "error", "message": "Prediction failed." }
    """
    data = request.get_json(silent=True) or {}
    student, error = validate_features(data)
    if error:
        return jsonify(error), 400

    try:
        result = full_prediction(student)
        return jsonify(result)
    except Exception:
        return jsonify({
            "status": "error",
            "message": "Prediction failed. Please check your input values.",
        }), 500


@app.route("/api/features", methods=["GET"])
def get_features():
    """
    Return feature metadata for dynamic form generation.

    The frontend should call this once on page load to build the input form.
    Each feature includes: display_name, description, min, max, step, unit,
    and direction (whether higher or lower is better).

    Response: {
        "features": [
            {
                "name": "Diet_Quality",
                "display_name": "Diet Quality",
                "description": "Self-rated overall diet quality...",
                "min": 1.0,
                "max": 4.0,
                "step": 0.5,
                "unit": "",
                "direction": "up"
            },
            ...
        ],
        "count": 20
    }
    """
    features = []
    for feat in FEATURE_ORDER:
        meta = FEATURE_METADATA[feat].copy()
        meta["name"] = feat
        features.append(meta)

    return jsonify({"features": features, "count": len(features)})


@app.route("/api/classes", methods=["GET"])
def get_classes():
    """
    Return the possible prediction class labels.

    Response: {
        "classes": ["At Risk", "Moderate", "Excellent"]
    }
    """
    return jsonify({"classes": list(CLASS_NAMES.values())})


@app.route("/api/sample-students", methods=["GET"])
def get_sample_students():
    """
    Return pre-filled sample student profiles for demo/testing.

    The frontend can offer a "Try a sample" dropdown that populates the form
    with one of these profiles so users can see the system in action.

    Response: {
        "samples": {
            "excellent": { "label": "...", "description": "...", "features": {...} },
            "moderate":  { ... },
            "at_risk":   { ... }
        }
    }
    """
    return jsonify({"samples": SAMPLE_STUDENTS})


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Academic Predictor Backend...")
    print(f"  Port: {PORT} | Debug: {DEBUG} | CORS: {CORS_ORIGINS}")
    print("  Endpoints:")
    print("    GET  /api/health          — Health check")
    print("    POST /api/predict         — Run prediction")
    print("    GET  /api/features        — Feature metadata")
    print("    GET  /api/classes         — Class labels")
    print("    GET  /api/sample-students — Sample profiles")
    print()
    # host=0.0.0.0 so hosting platforms can route external traffic to the app
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
