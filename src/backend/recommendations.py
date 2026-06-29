"""
recommendations.py — Per-feature feedback engine.

Generates structured, JSON-serializable recommendations based on a student's
feature values and their predicted performance tier. Ported from
src/early_warning_system.py but returns dicts instead of print strings.

FRONTEND CONTRACT:
    generate_recommendations() returns a list of dicts, each with:
      - feature       : internal feature name (e.g. "Stress_Level")
      - display_name  : human-friendly name (e.g. "Stress Level")
      - value         : the student's actual value
      - unit          : display unit (e.g. "/10", "hrs", or "")
      - severity      : one of "warning" | "improve" | "good" | "watch"
      - message       : actionable advice string

    compute_feature_profile() returns a list of dicts, each with:
      - feature    : internal feature name
      - value      : the student's actual value
      - percentage : value as % of the valid range (0-100)
      - min        : range minimum
      - max        : range maximum
"""

from feature_config import FEATURE_RANGES

# ── Advice lookup table ──────────────────────────────────────────────────────
# For each feature:
#   display_name   — human label
#   exc_thresh     — threshold for "Excellent" tier benchmark
#   risk_thresh    — threshold below/above which the student is "At Risk"
#   direction      — "up" = higher is better; "down" = lower is better
#   unit           — display unit
#   tip_good       — positive feedback for students meeting the benchmark
#   tip_bad        — actionable advice for students falling short
FEATURE_ADVICE = {
    "Diet_Quality": (
        "Diet Quality", 2.74, 2.05, "up", "",
        "Great diet habits — keep it up for sustained focus and energy.",
        "Poor diet is linked to lower cognitive performance. Aim for at least 3 balanced meals a day.",
    ),
    "Stress_Level": (
        "Stress Level", 5.69, 6.85, "down", "/10",
        "Stress is well-managed — maintain your current coping strategies.",
        "Dangerously high stress. Try time-boxing tasks, short breaks (Pomodoro), and sleep hygiene.",
    ),
    "AI_Tool_Usage": (
        "AI Tool Usage", 5.0, 2.0, "up", "",
        "Good AI tool adoption — keep using them critically for studying.",
        "Under-utilising AI tools. Try ChatGPT or Claude for concept explanations and practice questions.",
    ),
    "Total_Productive_Hrs": (
        "Productive Hours / Day", 5.14, 2.76, "up", "hrs",
        "Good productive hours. Protecting this time is key to your performance.",
        "Very low productive hours. Block 2-hr deep-work slots daily with phone out of reach.",
    ),
    "Distraction_Hrs": (
        "Distraction Hours / Day", 3.39, 5.96, "down", "hrs",
        "Distraction is under control — stay disciplined.",
        "Too many hours lost to distractions. Use app blockers (Forest, Cold Turkey) during study.",
    ),
    "Study_Distraction_Ratio": (
        "Study : Distraction Ratio", 1.32, 0.34, "up", "",
        "Solid focus ratio. Continue prioritising quality study time.",
        "Spending nearly as much time distracted as studying. Restructure your daily schedule.",
    ),
    "Wellbeing_Score": (
        "Wellbeing Score", 3.27, 0.29, "up", "",
        "Wellbeing is healthy — keep investing in rest and social support.",
        "Very low wellbeing. Speak to a counsellor or trusted peer; burnout risks performance long-term.",
    ),
    "Active_Engagement_Score": (
        "Active Engagement", 11.51, 8.68, "up", "",
        "High engagement is a strong predictor of success — keep participating actively.",
        "Low class/study engagement. Try active recall, group study, or office-hour visits.",
    ),
    "Study_Quality": (
        "Study Quality", 7.73, 2.69, "up", "",
        "Strong study quality. Maintain varied, spaced-repetition techniques.",
        "Study quality is low. Move from passive re-reading to practice problems and self-testing.",
    ),
    "AI_Enhanced_Study": (
        "AI-Enhanced Study", 9.79, 4.55, "up", "",
        "Good AI tool integration. Keep using them critically, not as a crutch.",
        "Under-utilising AI study tools. Try Claude, ChatGPT, or Khanmigo for concept explanations.",
    ),
    "Resource_Quality": (
        "Resource Quality", 7.40, 4.39, "up", "",
        "Access to quality resources is strong.",
        "Resource quality is weak. Visit your library, ask instructors for reading lists, use Khan Academy.",
    ),
    "Focus_Ratio": (
        "Focus Ratio", 2.11, 0.53, "up", "",
        "Good focus ratio — guard this with consistent sleep and environment.",
        "Focus ratio is critically low. Study in a quiet, dedicated space and reduce multitasking.",
    ),
    "Stress_Wellbeing": (
        "Stress-to-Wellbeing Ratio", 0.75, 1.83, "down", "",
        "Stress is balanced relative to wellbeing.",
        "Stress far outweighs wellbeing. Prioritise recovery: sleep, exercise, social connection.",
    ),
    "AI_Engagement": (
        "AI Engagement", 35.72, 21.91, "up", "",
        "Strong AI engagement score.",
        "Low AI engagement. Integrate AI tools into weekly study habits for explanations and feedback.",
    ),
    "Lifestyle_Index": (
        "Lifestyle Index", 0.56, 0.40, "up", "",
        "Healthy lifestyle balance.",
        "Lifestyle index is poor. Small wins count: 30-min walks, 7-8 hrs sleep, meal prepping.",
    ),
    "Study_Consistency": (
        "Study Consistency", 4.19, 1.63, "up", "",
        "Consistent study schedule — this is one of the biggest performance drivers.",
        "Irregular study pattern. Create a weekly timetable and treat study blocks like class time.",
    ),
    "Time_Efficiency": (
        "Time Efficiency", 1.29, 0.39, "up", "",
        "High time efficiency — you're getting good output per hour.",
        "Low time efficiency. Review HOW you study, not just how long. Active methods beat passive ones.",
    ),
    "Digital_Balance": (
        "Digital Balance", 1.25, 0.52, "up", "",
        "Good digital balance.",
        "Digital balance is off. Track screen time, set app limits, keep devices out of the bedroom.",
    ),
    "Academic_Momentum": (
        "Academic Momentum", 27.75, 13.87, "up", "",
        "Strong academic momentum — you're building on consistent effort.",
        "Academic momentum is stalling. Break goals into weekly milestones to rebuild progress.",
    ),
    "Stress_Load": (
        "Stress Load", 21.90, 45.46, "down", "",
        "Stress load is manageable.",
        "Extremely high stress load. Identify the top 2-3 stressors and tackle them one at a time.",
    ),
}


def generate_recommendations(student: dict, predicted_tier: int) -> list:
    """
    Generate per-feature recommendations based on the student's predicted tier.

    Parameters
    ----------
    student : dict
        The 20 base features from user input.
    predicted_tier : int
        0 = At Risk, 1 = Moderate, 2 = Excellent.

    Returns
    -------
    list[dict]
        Each dict has: feature, display_name, value, unit, severity, message.
        Severity values:
          - "warning"  : feature is past the risk threshold (urgent)
          - "improve"  : feature is between risk and excellence thresholds
          - "good"     : feature meets or exceeds the excellence benchmark
          - "watch"    : feature is slightly below excellent (minor concern)
    """
    results = []

    for feat, (name, exc_thresh, risk_thresh, direction, unit, tip_good, tip_bad) in FEATURE_ADVICE.items():
        val = student.get(feat)
        if val is None:
            continue

        entry = {
            "feature": feat,
            "display_name": name,
            "value": val,
            "unit": unit,
        }

        if predicted_tier == 0:
            # At Risk: only flag features past the risk threshold
            if direction == "up" and val < risk_thresh:
                entry["severity"] = "warning"
                entry["message"] = tip_bad
                results.append(entry)
            elif direction == "down" and val > risk_thresh:
                entry["severity"] = "warning"
                entry["message"] = tip_bad
                results.append(entry)

        elif predicted_tier == 1:
            # Moderate: flag warnings AND show improvement targets
            if direction == "up":
                if val < risk_thresh:
                    entry["severity"] = "warning"
                    entry["message"] = tip_bad
                    results.append(entry)
                elif val < exc_thresh:
                    entry["severity"] = "improve"
                    entry["message"] = f"Room to improve. Target > {exc_thresh}{unit}."
                    results.append(entry)
            else:
                if val > risk_thresh:
                    entry["severity"] = "warning"
                    entry["message"] = tip_bad
                    results.append(entry)
                elif val > exc_thresh:
                    entry["severity"] = "improve"
                    entry["message"] = f"Room to improve. Target < {exc_thresh}{unit}."
                    results.append(entry)

        else:
            # Excellent: confirmations + minor watch-outs
            if direction == "up" and val >= exc_thresh:
                entry["severity"] = "good"
                entry["message"] = tip_good
                results.append(entry)
            elif direction == "down" and val <= exc_thresh:
                entry["severity"] = "good"
                entry["message"] = tip_good
                results.append(entry)
            else:
                entry["severity"] = "watch"
                entry["message"] = (
                    f"Slightly below excellent benchmark ({exc_thresh}{unit}). Watch this."
                )
                results.append(entry)

    return results


def compute_feature_profile(student: dict) -> list:
    """
    Compute each feature's value as a percentage of its valid range.

    Used by the frontend to render horizontal bar charts showing where
    the student falls within the possible range for each feature.

    Parameters
    ----------
    student : dict
        The 20 base features from user input.

    Returns
    -------
    list[dict]
        Each dict has: feature, value, percentage (0-100), min, max.
    """
    profile = []
    for feat, (lo, hi) in FEATURE_RANGES.items():
        val = student.get(feat)
        if val is None:
            continue
        pct = max(0.0, min(100.0, (val - lo) / (hi - lo) * 100))
        profile.append({
            "feature": feat,
            "value": round(val, 2),
            "percentage": round(pct, 1),
            "min": lo,
            "max": hi,
        })
    return profile
