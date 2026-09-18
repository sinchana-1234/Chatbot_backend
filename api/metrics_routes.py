#!/usr/bin/env python3
"""
Patient Metrics API
Returns structured health summary as clean JSON — no agent, no LLM.
Supports: overall (all-time avg), weekly (last 7 days avg), daily (specific date),
          range (custom from_date to to_date avg)

Response structure (mobile-friendly):
  status, statusEmoji, statusMessage
  vitals[]        + vitalsNote
  activity[]      + activityNote
  recovery[]      + recoveryNote
  clinicalInsights[]
  actionPlan[]    + actionPlanFooter
"""

import logging
from typing import Optional
from datetime import date as date_cls, timedelta
from fastapi import APIRouter, Depends, Query, HTTPException

from auth.auth import get_current_user, UserContext
from dal.postgres_queries import get_mobile_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value, decimals=1):
    """Return rounded float as string, or None."""
    if value is None:
        return None
    try:
        return str(round(float(value), decimals))
    except:
        return None


def _fmt_int(value):
    """Return int as string, or None."""
    if value is None:
        return None
    try:
        return str(int(value))
    except:
        return None


def _fmt_sleep(minutes):
    """Return sleep as 'Xh Ym' string, or None."""
    if minutes is None:
        return None
    try:
        minutes = int(minutes)
        return f"{minutes // 60}h {minutes % 60}m"
    except:
        return None


def _fmt_bp(bp_str):
    """Return bp string '130/79', or None."""
    return bp_str if bp_str else None


# ---------------------------------------------------------------------------
# Status logic — ported directly from patient_summary_tool._get_overall_status
# ---------------------------------------------------------------------------

def _get_status(glucose, bp, spo2, stress):
    """
    Returns (status, statusEmoji, statusMessage).
    Mirrors the exact thresholds in patient_summary_tool.py.
    """
    if glucose is None and bp is None and spo2 is None and stress is None:
        return "No Data", "⚪", "No readings were recorded for this period."

    issues = 0
    if glucose and (glucose > 140 or glucose < 70):
        issues += 1
    if bp and "/" in str(bp):
        try:
            sys, dia = map(int, bp.split("/"))
            if sys > 130 or dia > 85:
                issues += 1
        except:
            pass
    if spo2 and spo2 < 95:
        issues += 1
    if stress and stress > 60:
        issues += 1

    if issues == 0:
        return "Excellent", "🟢", "Great job! You are consistently hitting your health targets."
    elif issues == 1:
        return "Good", "🟡", "Most of your readings are in range. Keep an eye on flagged metrics."
    else:
        return "Needs Attention", "🔴", "Some of your readings need attention. Please consult your healthcare provider."


# ---------------------------------------------------------------------------
# Clinical insights — ported from patient_summary_tool._get_condition_text
# Returns a list of strings instead of a single formatted string
# ---------------------------------------------------------------------------

def _get_clinical_insights(glucose, bp, spo2, stress):
    """
    Returns list of clinical insight strings.
    Mirrors the exact thresholds in patient_summary_tool.py.
    """
    if glucose is None and bp is None and spo2 is None and stress is None:
        return ["No clinical readings available for this period."]

    severe = []
    mild = []

    if glucose:
        if glucose > 180:
            severe.append(f"Glucose is high at {round(float(glucose), 1)} mg/dL (normal: 70-140). Consult your doctor.")
        elif glucose < 70:
            severe.append(f"Glucose is low at {round(float(glucose), 1)} mg/dL (normal: 70-140). Have a snack and monitor.")
        elif glucose > 140:
            mild.append(f"Glucose slightly elevated at {round(float(glucose), 1)} mg/dL. A short walk after meals can help.")

    if bp and "/" in str(bp):
        try:
            sys, dia = map(int, bp.split("/"))
            if sys > 140 or dia > 90:
                severe.append(f"Blood pressure high at {sys}/{dia} mmHg. Please consult your doctor.")
            elif sys > 130 or dia > 85:
                mild.append(f"Blood pressure slightly elevated at {sys}/{dia} mmHg. Reduce salt and stress.")
        except:
            pass

    if spo2 and spo2 < 95:
        severe.append(f"Oxygen level low at {round(float(spo2), 1)}%. Seek medical attention if breathless.")

    if stress:
        if stress > 80:
            severe.append(f"Stress very high ({round(float(stress), 1)}/100). Rest and consider speaking to someone.")
        elif stress > 60:
            mild.append(f"Stress is moderate ({round(float(stress), 1)}/100). Try breathing exercises or a short walk.")

    if severe:
        return severe
    if mild:
        return mild
    return ["All clinical parameters are within normal range."]


# ---------------------------------------------------------------------------
# Action plan — ported from patient_summary_tool hardcoded action plan section
# ---------------------------------------------------------------------------

def _get_action_plan():
    """
    Returns the same action plan as patient_summary_tool.py (lines 250-252).
    """
    return [
        {
            "title": "Prioritize Hydration",
            "description": "Aim for 2-3L of water daily to support metabolic function."
        },
        {
            "title": "Maintain Sleep Consistency",
            "description": "Keep a regular sleep schedule for optimal recovery."
        },
        {
            "title": "Stay Active",
            "description": "Even light daily movement helps improve insulin sensitivity."
        }
    ]


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

@router.get("/summary")
async def get_metrics_summary(
    period: Optional[str] = Query(
        default="overall",
        description="overall | weekly | daily | range"
    ),
    date: Optional[str] = Query(
        default=None,
        description="YYYY-MM-DD — required when period=daily"
    ),
    from_date: Optional[str] = Query(
        default=None,
        description="YYYY-MM-DD — start of range, required when period=range"
    ),
    to_date: Optional[str] = Query(
        default=None,
        description="YYYY-MM-DD — end of range, required when period=range"
    ),
    current_user: UserContext = Depends(get_current_user)
):
    """
    Returns a structured health summary for the authenticated patient.

    - period=overall                            -> all-time averages
    - period=weekly                             -> last 7 days averages
    - period=daily&date=YYYY-MM-DD              -> specific date averages
    - period=range&from_date=X&to_date=Y        -> custom date range averages
    """
    patient_id = current_user.user_id

    target_date = None
    start_date  = None
    end_date    = None

    if period == "overall":
        pass  # No date filter — all-time averages

    elif period == "weekly":
        start_date = (date_cls.today() - timedelta(days=6)).strftime("%Y-%m-%d")

    elif period == "daily":
        if date:
            try:
                date_cls.fromisoformat(date)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid date format '{date}'. Expected YYYY-MM-DD (e.g. 2026-05-22)."
                )
            target_date = date
        else:
            target_date = (date_cls.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    elif period == "range":
        if not from_date or not to_date:
            raise HTTPException(
                status_code=400,
                detail="period=range requires both from_date and to_date (YYYY-MM-DD)."
            )
        try:
            from_dt = date_cls.fromisoformat(from_date)
            to_dt   = date_cls.fromisoformat(to_date)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format. Expected YYYY-MM-DD for from_date and to_date."
            )
        if from_dt > to_dt:
            raise HTTPException(
                status_code=400,
                detail=f"from_date ({from_date}) cannot be after to_date ({to_date})."
            )
        start_date = from_date
        end_date   = to_date

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid period. Use: overall | weekly | daily | range"
        )

    try:
        data = get_mobile_metrics(patient_id, target_date, start_date, end_date)
    except Exception as e:
        logger.error(f"Error fetching metrics for patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch metrics.")

    if not data:
        raise HTTPException(status_code=404, detail="No health data found.")

    # Pull raw values
    glucose  = data.get("glucose")
    bp       = data.get("bp")
    temp     = data.get("temperature")
    spo2     = data.get("spo2")
    hr       = data.get("heart_rate")
    hrv      = data.get("hrv")
    stress   = data.get("stress")
    sleep    = data.get("sleep")
    steps    = data.get("steps")
    calories = data.get("calories_burned")
    active   = data.get("active_time")

    # Compute status, insights
    status, status_emoji, status_message = _get_status(glucose, bp, spo2, stress)
    clinical_insights = _get_clinical_insights(glucose, bp, spo2, stress)
    # action_plan, *Note, and actionPlanFooter are intentionally None below.
    # The previous implementation returned hardcoded text claiming to be
    # "Personalized" advice and patient-specific notes — but the content was
    # identical for every patient regardless of their metrics. Returning None
    # is the honest signal until a real dynamic implementation is built.
    # Frontend should hide these sections when the field is null.

    return {
        "patient_id": patient_id,
        "period": period,
        "from_date": from_date if period == "range" else None,
        "to_date": to_date if period == "range" else None,

        # --- Overall Status ---
        "status": status,
        "statusEmoji": status_emoji,
        "statusMessage": status_message, 

        # --- Vitals group (from summary: Vital Signs at a Glance) ---
        "vitals": [
            {
                "icon": "🩸",
                "name": "Glucose",
                "value": _fmt(glucose),
                "unit": "mg/dL"
            },
            {
                "icon": "💊",
                "name": "Blood Pressure",
                "value": _fmt_bp(bp),
                "unit": "mmHg"
            },
            {
                "icon": "❤️",
                "name": "Heart Rate",
                "value": _fmt(hr),
                "unit": "bpm"
            }
        ],
        # Was a hardcoded generic note. Hidden until dynamic version ships.
        "vitalsNote": None,

        # --- Activity group (from summary: Activity & Movement) ---
        "activity": [
            {
                "icon": "🚶",
                "name": "Steps",
                "value": _fmt_int(steps),
                "unit": "steps"
            },
            {
                "icon": "🔥",
                "name": "Calories Burned",
                "value": _fmt(calories),
                "unit": "kcal"
            },
            {
                "icon": "⏱️",
                "name": "Active Time",
                "value": _fmt_int(active),
                "unit": "mins"
            }
        ],
        # Was a hardcoded generic note. Hidden until dynamic version ships.
        "activityNote": None,

        # --- Recovery group (from summary: Recovery & Wellness) ---
        "recovery": [
            {
                "icon": "💤",
                "name": "Average Sleep",
                "value": _fmt_sleep(sleep),
                # "unit": "hrs"
            },
            {
                "icon": "📈",
                "name": "HRV",
                "value": _fmt(hrv),
                "unit": "ms"
            },
            {
                "icon": "🧠",
                "name": "Stress Level",
                "value": _fmt(stress),
                "unit": "/100"
            }
        ],
        # Was a hardcoded generic note. Hidden until dynamic version ships.
        "recoveryNote": None,

        # --- Clinical Insights (from summary: Clinical Insights & Analysis) ---
        "clinicalInsights": clinical_insights,

        # --- Action Plan: was hardcoded 3 items returned to every patient
        #     regardless of their metrics. Hidden until dynamic version ships.
        "actionPlan": None,
        "actionPlanFooter": None,
    }