"""
feature_config.py — Feature metadata, validation ranges, and constants.

This file is the single source of truth for feature definitions used by both
the backend prediction logic and (indirectly) the frontend form generation.
The GET /api/features endpoint returns data derived from these structures.

FRONTEND CONTRACT:
    - FEATURE_ORDER lists the 20 input features in the exact order the
      classification model expects them.
    - FEATURE_METADATA provides everything needed to render an input form:
      display_name, description, min, max, step, unit, and direction.
    - GPA_REGRESSION_FEATURES lists the 10 features (6 base + 4 engineered)
      used by the GPA regression model. The frontend does NOT need to send
      these separately — the backend computes derived features automatically
      from the 20 base features.
"""

# ── Classification model: 20 input features in trained order ─────────────────
# These are the exact features selected by SelectKBest(mutual_info_classif, k=20)
# during training. The order matters — the scaler and model expect this sequence.
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

# ── GPA regression model: 10 features (6 base + 4 engineered) ───────────────
# The backend computes the 4 engineered features from the user's base inputs.
# See feature_engineering.py for the computation logic.
GPA_REGRESSION_FEATURES = [
    "Active_Engagement_Score",    # base
    "Study_Quality",              # base
    "Resource_Quality",           # base
    "Lifestyle_Index",            # base
    "Academic_Momentum",          # base
    "Stress_Load",                # base
    "Quality_Consistency",        # engineered: Study_Quality * Study_Consistency
    "AI_Leverage",                # engineered: AI_Enhanced_Study * AI_Engagement
    "Learning_Power",             # engineered: Resource_Quality * AI_Enhanced_Study * Study_Consistency
    "Stress_Penalty",             # engineered: Stress_Load * (1 - Lifestyle_Index)
]

# ── Class labels for tier classification ─────────────────────────────────────
CLASS_NAMES = {0: "At Risk", 1: "Moderate", 2: "Excellent"}

# ── Feature valid ranges (min, max from dataset) ────────────────────────────
# Used for input validation and frontend slider/bar normalization.
FEATURE_RANGES = {
    "Diet_Quality":            (1.0,    4.0),
    "Stress_Level":            (1.5,    9.5),
    "AI_Tool_Usage":           (0.0,   10.0),
    "Total_Productive_Hrs":    (0.0,   16.0),
    "Distraction_Hrs":         (0.5,   10.0),
    "Study_Distraction_Ratio": (0.0,    8.33),
    "Wellbeing_Score":         (-5.5,  10.5),
    "Active_Engagement_Score": (5.5,   13.5),
    "Study_Quality":           (0.0,   18.0),
    "AI_Enhanced_Study":       (0.0,   24.0),
    "Resource_Quality":        (1.0,   16.0),
    "Focus_Ratio":             (0.0,   20.0),
    "Stress_Wellbeing":        (0.09,  19.0),
    "AI_Engagement":           (0.0,   54.0),
    "Lifestyle_Index":         (0.17,   0.97),
    "Study_Consistency":       (0.0,    9.0),
    "Time_Efficiency":         (0.0,    7.2),
    "Digital_Balance":         (0.0,    6.67),
    "Academic_Momentum":       (5.5,   40.5),
    "Stress_Load":             (2.44,  96.19),
}

# ── Feature metadata for frontend form generation ────────────────────────────
# Each feature has:
#   display_name  — human-friendly label for the UI
#   description   — tooltip/helper text explaining what this feature measures
#   min, max      — valid input range (from dataset)
#   step          — recommended slider/input step increment
#   unit          — display unit (empty string if dimensionless)
#   direction     — "up" means higher is better, "down" means lower is better
FEATURE_METADATA = {
    "Diet_Quality": {
        "display_name": "Diet Quality",
        "description": "Self-rated overall diet quality (1 = poor, 4 = excellent)",
        "min": 1.0, "max": 4.0, "step": 0.5,
        "unit": "", "direction": "up",
    },
    "Stress_Level": {
        "display_name": "Stress Level",
        "description": "Academic stress level on a 1-10 scale",
        "min": 1.5, "max": 9.5, "step": 0.5,
        "unit": "/10", "direction": "down",
    },
    "AI_Tool_Usage": {
        "display_name": "AI Tool Usage",
        "description": "Frequency of AI tool use for studying (0-10 scale)",
        "min": 0.0, "max": 10.0, "step": 1.0,
        "unit": "", "direction": "up",
    },
    "Total_Productive_Hrs": {
        "display_name": "Productive Hours / Day",
        "description": "Total productive study hours per day",
        "min": 0.0, "max": 16.0, "step": 0.5,
        "unit": "hrs", "direction": "up",
    },
    "Distraction_Hrs": {
        "display_name": "Distraction Hours / Day",
        "description": "Hours lost to distractions (social media, gaming, etc.) per day",
        "min": 0.5, "max": 10.0, "step": 0.5,
        "unit": "hrs", "direction": "down",
    },
    "Study_Distraction_Ratio": {
        "display_name": "Study : Distraction Ratio",
        "description": "Ratio of study time to distraction time (higher = more focused)",
        "min": 0.0, "max": 8.33, "step": 0.1,
        "unit": "", "direction": "up",
    },
    "Wellbeing_Score": {
        "display_name": "Wellbeing Score",
        "description": "Composite wellbeing score combining sleep, social support, and mental health",
        "min": -5.5, "max": 10.5, "step": 0.5,
        "unit": "", "direction": "up",
    },
    "Active_Engagement_Score": {
        "display_name": "Active Engagement",
        "description": "Class participation and active study engagement score",
        "min": 5.5, "max": 13.5, "step": 0.5,
        "unit": "", "direction": "up",
    },
    "Study_Quality": {
        "display_name": "Study Quality",
        "description": "Perceived quality of study sessions (active recall, practice problems, etc.)",
        "min": 0.0, "max": 18.0, "step": 0.5,
        "unit": "", "direction": "up",
    },
    "AI_Enhanced_Study": {
        "display_name": "AI-Enhanced Study",
        "description": "Degree to which AI tools enhance your learning",
        "min": 0.0, "max": 24.0, "step": 1.0,
        "unit": "", "direction": "up",
    },
    "Resource_Quality": {
        "display_name": "Resource Quality",
        "description": "Quality of study resources (textbooks, online materials, library access)",
        "min": 1.0, "max": 16.0, "step": 1.0,
        "unit": "", "direction": "up",
    },
    "Focus_Ratio": {
        "display_name": "Focus Ratio",
        "description": "Ratio of focused study time to total time spent studying",
        "min": 0.0, "max": 20.0, "step": 0.1,
        "unit": "", "direction": "up",
    },
    "Stress_Wellbeing": {
        "display_name": "Stress-to-Wellbeing Ratio",
        "description": "Ratio of stress to wellbeing — lower means stress is better managed",
        "min": 0.09, "max": 19.0, "step": 0.1,
        "unit": "", "direction": "down",
    },
    "AI_Engagement": {
        "display_name": "AI Engagement",
        "description": "Combined score of AI tool usage frequency and depth of engagement",
        "min": 0.0, "max": 54.0, "step": 1.0,
        "unit": "", "direction": "up",
    },
    "Lifestyle_Index": {
        "display_name": "Lifestyle Index",
        "description": "Composite lifestyle health (diet, exercise, sleep, social balance)",
        "min": 0.17, "max": 0.97, "step": 0.01,
        "unit": "", "direction": "up",
    },
    "Study_Consistency": {
        "display_name": "Study Consistency",
        "description": "Regularity of study schedule (0 = erratic, 9 = very consistent)",
        "min": 0.0, "max": 9.0, "step": 0.5,
        "unit": "", "direction": "up",
    },
    "Time_Efficiency": {
        "display_name": "Time Efficiency",
        "description": "Productive output per hour of study time",
        "min": 0.0, "max": 7.2, "step": 0.1,
        "unit": "", "direction": "up",
    },
    "Digital_Balance": {
        "display_name": "Digital Balance",
        "description": "Balance between productive digital tool use and distracting screen time",
        "min": 0.0, "max": 6.67, "step": 0.1,
        "unit": "", "direction": "up",
    },
    "Academic_Momentum": {
        "display_name": "Academic Momentum",
        "description": "Trend in academic performance — are grades improving or declining?",
        "min": 5.5, "max": 40.5, "step": 0.5,
        "unit": "", "direction": "up",
    },
    "Stress_Load": {
        "display_name": "Stress Load",
        "description": "Combined academic and personal stress workload index",
        "min": 2.44, "max": 96.19, "step": 1.0,
        "unit": "", "direction": "down",
    },
}

# ── GPA to letter grade mapping ──────────────────────────────────────────────
# Standard 4.0 scale used by most Cambodian universities.
def gpa_to_grade(gpa: float) -> str:
    """Convert a numeric GPA to a letter grade."""
    if gpa >= 3.7:
        return "A"
    elif gpa >= 3.3:
        return "A-"
    elif gpa >= 3.0:
        return "B+"
    elif gpa >= 2.7:
        return "B"
    elif gpa >= 2.3:
        return "B-"
    elif gpa >= 2.0:
        return "C+"
    elif gpa >= 1.7:
        return "C"
    elif gpa >= 1.3:
        return "D"
    else:
        return "F"


# ── Sample student profiles for demo/testing ─────────────────────────────────
# These profiles represent typical students in each tier.
# Frontend can fetch these via GET /api/sample-students to pre-fill the form.
SAMPLE_STUDENTS = {
    "excellent": {
        "label": "Excellent Student",
        "description": "High-performing student with strong study habits and low stress",
        "features": {
            "Diet_Quality": 3.0,
            "Stress_Level": 5.0,
            "AI_Tool_Usage": 6.0,
            "Total_Productive_Hrs": 8.0,
            "Distraction_Hrs": 2.0,
            "Study_Distraction_Ratio": 2.5,
            "Wellbeing_Score": 6.0,
            "Active_Engagement_Score": 10.5,
            "Study_Quality": 15.0,
            "AI_Enhanced_Study": 15.0,
            "Resource_Quality": 10.0,
            "Focus_Ratio": 4.0,
            "Stress_Wellbeing": 0.3,
            "AI_Engagement": 45.0,
            "Lifestyle_Index": 0.8,
            "Study_Consistency": 7.0,
            "Time_Efficiency": 3.0,
            "Digital_Balance": 2.0,
            "Academic_Momentum": 30.0,
            "Stress_Load": 15.0,
        },
    },
    "moderate": {
        "label": "Moderate Student",
        "description": "Average student with room for improvement in several areas",
        "features": {
            "Diet_Quality": 2.0,
            "Stress_Level": 6.2,
            "AI_Tool_Usage": 4.0,
            "Total_Productive_Hrs": 3.5,
            "Distraction_Hrs": 4.5,
            "Study_Distraction_Ratio": 0.60,
            "Wellbeing_Score": 1.8,
            "Active_Engagement_Score": 10.5,
            "Study_Quality": 4.0,
            "AI_Enhanced_Study": 6.5,
            "Resource_Quality": 5.0,
            "Focus_Ratio": 1.00,
            "Stress_Wellbeing": 1.00,
            "AI_Engagement": 30.0,
            "Lifestyle_Index": 0.43,
            "Study_Consistency": 2.8,
            "Time_Efficiency": 0.63,
            "Digital_Balance": 0.85,
            "Academic_Momentum": 20.0,
            "Stress_Load": 31.0,
        },
    },
    "at_risk": {
        "label": "At-Risk Student",
        "description": "Student showing warning signs: high stress, low engagement, poor habits",
        "features": {
            "Diet_Quality": 1.0,
            "Stress_Level": 8.5,
            "AI_Tool_Usage": 1.0,
            "Total_Productive_Hrs": 1.0,
            "Distraction_Hrs": 8.0,
            "Study_Distraction_Ratio": 0.10,
            "Wellbeing_Score": -2.0,
            "Active_Engagement_Score": 6.0,
            "Study_Quality": 1.0,
            "AI_Enhanced_Study": 1.0,
            "Resource_Quality": 2.0,
            "Focus_Ratio": 0.15,
            "Stress_Wellbeing": 3.50,
            "AI_Engagement": 5.0,
            "Lifestyle_Index": 0.20,
            "Study_Consistency": 0.0,
            "Time_Efficiency": 0.10,
            "Digital_Balance": 0.10,
            "Academic_Momentum": 6.0,
            "Stress_Load": 75.0,
        },
    },
}
