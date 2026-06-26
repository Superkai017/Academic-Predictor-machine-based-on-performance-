"""
api.py — FastAPI backend for Academic Performance Predictor.

Run with:
    uvicorn api:app --reload

Docs available at:
    http://localhost:8000/docs
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Load artifacts ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model  = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
except FileNotFoundError as e:
    raise RuntimeError(f"Model artifact not found: {e}")


# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_ORDER = [
    "Diet_Quality",
    "Stress_Level",
    "AI_Tool_Usage",
    "Total_Productive_Hrs",
    "Distraction_Hrs",
    "Study_Distraction_Ratio",
    "Wellbeing_Score",
    "Active_Engagement_Score",
    "Study_Quality",
    "AI_Enhanced_Study",
    "Resource_Quality",
    "Focus_Ratio",
    "Stress_Wellbeing",
    "AI_Engagement",
    "Lifestyle_Index",
    "Study_Consistency",
    "Time_Efficiency",
    "Digital_Balance",
    "Academic_Momentum",
    "Stress_Load",
]

CLASS_LABELS = {0: "At Risk", 1: "Moderate", 2: "Excellent"}

ADVICE = {
    "At Risk": (
        "Your performance indicates you may need additional support. "
        "Consider reducing distractions, improving sleep and diet, "
        "and reaching out to academic advisors."
    ),
    "Moderate": (
        "You are on a decent track. Focus on consistency, "
        "manage stress better, and leverage AI tools more effectively "
        "to push toward Excellent."
    ),
    "Excellent": (
        "Great work! Keep maintaining your study habits, "
        "wellbeing, and engagement levels."
    ),
}


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Academic Performance Predictor API",
    description="Predicts student performance tier (At Risk / Moderate / Excellent) based on lifestyle and study features.",
    version="1.0.0",
)

# Allow all origins for demo — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schema ────────────────────────────────────────────────────────────
class StudentFeatures(BaseModel):
    Diet_Quality:             float = Field(..., ge=0, le=10,  description="Diet quality score (0–10)",                  json_schema_extra={"example": 7.0})
    Stress_Level:             float = Field(..., ge=0, le=10,  description="Stress level (0–10)",                        json_schema_extra={"example": 5.0})
    AI_Tool_Usage:            float = Field(..., ge=0, le=10,  description="Frequency of AI tool usage (0–10)",          json_schema_extra={"example": 6.0})
    Total_Productive_Hrs:     float = Field(..., ge=0,         description="Total productive hours per day",             json_schema_extra={"example": 5.0})
    Distraction_Hrs:          float = Field(..., ge=0,         description="Hours lost to distractions per day",         json_schema_extra={"example": 2.0})
    Study_Distraction_Ratio:  float = Field(..., ge=0,         description="Ratio of study to distraction hours",        json_schema_extra={"example": 2.5})
    Wellbeing_Score:          float = Field(..., ge=0, le=10,  description="Overall wellbeing score (0–10)",             json_schema_extra={"example": 7.0})
    Active_Engagement_Score:  float = Field(..., ge=0, le=10,  description="Active class/study engagement (0–10)",       json_schema_extra={"example": 6.5})
    Study_Quality:            float = Field(..., ge=0, le=10,  description="Perceived study quality (0–10)",             json_schema_extra={"example": 7.0})
    AI_Enhanced_Study:        float = Field(..., ge=0, le=10,  description="Degree AI tools enhance study (0–10)",       json_schema_extra={"example": 5.0})
    Resource_Quality:         float = Field(..., ge=0, le=10,  description="Quality of study resources (0–10)",          json_schema_extra={"example": 8.0})
    Focus_Ratio:              float = Field(..., ge=0,         description="Ratio of focused to total study time",       json_schema_extra={"example": 0.75})
    Stress_Wellbeing:         float = Field(...,               description="Interaction of stress and wellbeing",        json_schema_extra={"example": 0.7})
    AI_Engagement:            float = Field(..., ge=0,         description="Combined AI usage and engagement score",     json_schema_extra={"example": 5.5})
    Lifestyle_Index:          float = Field(...,               description="Composite lifestyle health index",           json_schema_extra={"example": 6.0})
    Study_Consistency:        float = Field(..., ge=0, le=10,  description="Consistency of study schedule (0–10)",       json_schema_extra={"example": 7.0})
    Time_Efficiency:          float = Field(..., ge=0,         description="Ratio of productive to total available hrs", json_schema_extra={"example": 0.8})
    Digital_Balance:          float = Field(...,               description="Balance between digital tool use and focus", json_schema_extra={"example": 5.0})
    Academic_Momentum:        float = Field(...,               description="Trend in academic performance over time",    json_schema_extra={"example": 6.5})
    Stress_Load:              float = Field(..., ge=0,         description="Combined stress and workload index",         json_schema_extra={"example": 4.0})


# ── Response schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction:   str
    confidence:   float
    probabilities: dict[str, float]
    advice:       str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the API is running."""
    return {"status": "ok", "message": "Academic Performance Predictor API is running."}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: StudentFeatures):
    """
    Predict a student's performance tier from their feature values.

    Returns the predicted class, confidence, per-class probabilities, and advice.
    """
    try:
        # Build feature vector in the exact order the model was trained on
        X = np.array([[getattr(data, feat) for feat in FEATURE_ORDER]])

        X_scaled  = scaler.transform(X)
        pred_idx  = int(model.predict(X_scaled)[0])
        proba     = model.predict_proba(X_scaled)[0]

        label = CLASS_LABELS[pred_idx]

        return PredictionResponse(
            prediction=label,
            confidence=round(float(proba[pred_idx]), 4),
            probabilities={
                CLASS_LABELS[i]: round(float(p), 4)
                for i, p in enumerate(proba)
            },
            advice=ADVICE[label],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features", tags=["Meta"])
def get_features():
    """Returns the list of expected input features in order."""
    return {"features": FEATURE_ORDER, "count": len(FEATURE_ORDER)}


@app.get("/classes", tags=["Meta"])
def get_classes():
    """Returns the possible prediction classes."""
    return {"classes": list(CLASS_LABELS.values())}