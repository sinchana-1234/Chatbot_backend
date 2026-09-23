#!/usr/bin/env python3
"""
Correlation Analysis Tool for Revival Medical System

Analyzes how stress, sleep, and activity correlate with patient's glucose patterns.
Uses the favorability table logic:
- Stress: 0-60 favorable, 61-100 unfavorable
- Sleep: 7-9 hours favorable, <7 or >9 hours unfavorable
- Activity: ≥27,000 steps favorable, <4,000 steps unfavorable
- Returns Overall Association count (2-3 favorable = good, mixed = neutral, etc.)
"""

import json
import logging
from typing import Optional
from datetime import datetime, timedelta
from langchain.tools import BaseTool
from dal.database import DatabaseManager

logger = logging.getLogger(__name__)


class CorrelationAnalysisTool(BaseTool):
    """Analyzes correlations between stress/sleep/activity and glucose patterns"""

    name: str = "analyze_glucose_correlations"
    description: str = """Analyze how stress, sleep, and activity correlate with a patient's glucose patterns.

    Parameters:
    - patient_id (int): Patient ID (optional for patient role, required for staff)
    - patient_name (str): Patient name (alternative to patient_id)
    - from_date (str): Start date (YYYY-MM-DD) — OPTIONAL
    - to_date (str): End date (YYYY-MM-DD) — OPTIONAL
    - period (str): OPTIONAL relative phrase, e.g. 'last 30 days', 'this month'

    Returns: Overall Association summary showing:
    - Stress pattern favorability (0-60 = favorable, 61-100 = unfavorable)
    - Sleep pattern favorability (7-9 hours = favorable, else unfavorable)
    - Activity pattern favorability (≥27,000 steps = favorable, <4,000 = unfavorable)
    - Overall Association (2-3 favorable patterns = good, mixed = neutral, etc.)
    - Main concerns based on patterns

    Use for questions like:
    - "How does activity affect this patient's glucose?"
    - "How do sleep and stress relate to glucose?"
    - "What are the main concerns in this patient's data?"
    - "How has this patient's glucose changed over time?"
    """

    def __init__(self):
        super().__init__()

    def set_user_context(self, user_context):
        object.__setattr__(self, 'user_context', user_context)

    def _resolve_patient_id(self, patient_id: Optional[int], patient_name: Optional[str]) -> tuple:
        """Resolve patient name to ID if needed"""
        if patient_id:
            return patient_id, None
        if not patient_name:
            return None, json.dumps({
                "status": "not_found",
                "notice": "Either patient_id or patient_name is required."
            })

        user_context = getattr(self, 'user_context', None)

        with DatabaseManager() as db_manager:
            doctor_id = user_context.get('user_id') if user_context else None
            own_patients = db_manager.get_doctor_patients(doctor_user_id=doctor_id) if doctor_id else []
            own_matching = [
                p for p in own_patients
                if patient_name.lower() in f"{p.get('patient_first_name') or ''} {p.get('patient_last_name') or ''}".lower()
            ]

            if own_matching:
                if len(own_matching) > 1:
                    return None, json.dumps({
                        "status": "ambiguous_name",
                        "notice": f"Multiple patients match '{patient_name}'. Ask the user which one.",
                        "matching_patients": [
                            {
                                "id": p["patient_id"],
                                "name": f"{p.get('patient_first_name') or ''} {p.get('patient_last_name') or ''}".strip(),
                                "email": p.get("patient_email")
                            } for p in own_matching
                        ]
                    })
                return own_matching[0]["patient_id"], None

            users = db_manager.get_users()
            matching_users = [
                u for u in users
                if patient_name.lower() in f"{u.first_name or ''} {u.last_name or ''}".lower()
                and u.role_id == 1
            ]

            if not matching_users:
                return None, json.dumps({
                    "status": "not_found",
                    "notice": f"No patient found with name containing '{patient_name}'."
                })

            if len(matching_users) > 1:
                return None, json.dumps({
                    "status": "ambiguous_name",
                    "notice": f"Multiple patients match '{patient_name}'. Ask the user which one.",
                    "matching_patients": [
                        {
                            "id": u.id,
                            "name": f"{u.first_name or ''} {u.last_name or ''}".strip(),
                            "email": u.email
                        } for u in matching_users
                    ]
                })

            return matching_users[0].id, None

    def _analyze_stress_pattern(self, stress_data: dict) -> dict:
        """
        Analyze stress favorability
        Favorable: 0-60
        Unfavorable: 61-100
        """
        if not stress_data or stress_data.get("days_count", 0) == 0:
            return {
                "status": "no_data",
                "favorability": "unknown",
                "avg_stress": None
            }

        avg_stress = stress_data.get("avg_stress_pct", 0)

        if avg_stress is None:
            return {
                "status": "no_data",
                "favorability": "unknown",
                "avg_stress": None
            }

        favorability = "favorable" if avg_stress <= 60 else "unfavorable"

        return {
            "status": "ok",
            "favorability": favorability,
            "avg_stress": avg_stress,
            "interpretation": f"Average stress: {avg_stress}% ({'low/healthy' if avg_stress <= 60 else 'high/concerning'})"
        }

    def _analyze_sleep_pattern(self, sleep_data: dict) -> dict:
        """
        Analyze sleep favorability
        Favorable: 7-9 hours
        Unfavorable: <7 or >9 hours
        """
        if not sleep_data or sleep_data.get("days_count", 0) == 0:
            return {
                "status": "no_data",
                "favorability": "unknown",
                "avg_sleep_hours": None
            }

        avg_light = sleep_data.get("avg_light_hrs", 0) or 0
        avg_deep = sleep_data.get("avg_deep_hrs", 0) or 0
        avg_rem = sleep_data.get("avg_rem_hrs", 0) or 0
        total_sleep = avg_light + avg_deep + avg_rem

        if total_sleep == 0:
            return {
                "status": "no_data",
                "favorability": "unknown",
                "avg_sleep_hours": None
            }

        favorability = "favorable" if 7 <= total_sleep <= 9 else "unfavorable"

        return {
            "status": "ok",
            "favorability": favorability,
            "avg_sleep_hours": round(total_sleep, 1),
            "breakdown": {
                "deep_hrs": round(avg_deep, 1),
                "light_hrs": round(avg_light, 1),
                "rem_hrs": round(avg_rem, 1)
            },
            "interpretation": f"Average sleep: {round(total_sleep, 1)} hours ({'optimal' if 7 <= total_sleep <= 9 else 'suboptimal'})"
        }

    def _analyze_activity_pattern(self, activity_data: dict) -> dict:
        """
        Analyze activity favorability
        Favorable: ≥27,000 steps
        Unfavorable: <4,000 steps
        Neutral: 4,000-27,000 steps
        """
        if not activity_data or activity_data.get("days_count", 0) == 0:
            return {
                "status": "no_data",
                "favorability": "unknown",
                "avg_steps": None
            }

        avg_steps = activity_data.get("avg_steps", 0)

        if avg_steps is None or avg_steps == 0:
            return {
                "status": "no_data",
                "favorability": "unknown",
                "avg_steps": None
            }

        if avg_steps >= 27000:
            favorability = "favorable"
            interpretation = "High activity level (excellent for glucose control)"
        elif avg_steps < 4000:
            favorability = "unfavorable"
            interpretation = "Low activity level (concerning for glucose control)"
        else:
            favorability = "neutral"
            interpretation = "Moderate activity level"

        return {
            "status": "ok",
            "favorability": favorability,
            "avg_steps": int(avg_steps),
            "interpretation": interpretation
        }

    def _calculate_overall_association(self, stress: dict, sleep: dict, activity: dict) -> dict:
        """
        Calculate overall association based on favorability counts:
        - 2-3 favorable patterns = good association
        - 1 favorable + 1-2 unfavorable = mixed association
        - 0-1 favorable = concerning association
        """
        favorable_count = 0
        unfavorable_count = 0
        unknown_count = 0

        patterns = []

        # Count stress
        if stress.get("status") == "ok":
            if stress.get("favorability") == "favorable":
                favorable_count += 1
                patterns.append(f"✅ Low stress ({stress.get('avg_stress')}%)")
            elif stress.get("favorability") == "unfavorable":
                unfavorable_count += 1
                patterns.append(f"❌ High stress ({stress.get('avg_stress')}%)")
        else:
            unknown_count += 1
            patterns.append("⚠️ Stress data: insufficient")

        # Count sleep
        if sleep.get("status") == "ok":
            if sleep.get("favorability") == "favorable":
                favorable_count += 1
                patterns.append(f"✅ Optimal sleep ({sleep.get('avg_sleep_hours')} hrs)")
            elif sleep.get("favorability") == "unfavorable":
                unfavorable_count += 1
                patterns.append(f"❌ Suboptimal sleep ({sleep.get('avg_sleep_hours')} hrs)")
        else:
            unknown_count += 1
            patterns.append("⚠️ Sleep data: insufficient")

        # Count activity
        if activity.get("status") == "ok":
            if activity.get("favorability") == "favorable":
                favorable_count += 1
                patterns.append(f"✅ High activity ({activity.get('avg_steps'):,} steps)")
            elif activity.get("favorability") == "unfavorable":
                unfavorable_count += 1
                patterns.append(f"❌ Low activity ({activity.get('avg_steps'):,} steps)")
            else:  # neutral
                patterns.append(f"⚠️ Moderate activity ({activity.get('avg_steps'):,} steps)")
        else:
            unknown_count += 1
            patterns.append("⚠️ Activity data: insufficient")

        # Determine overall association
        if favorable_count >= 2:
            overall = "Good - Multiple favorable patterns supporting glucose control"
        elif favorable_count == 1 and unfavorable_count <= 1:
            overall = "Mixed - Some favorable, some unfavorable patterns"
        elif favorable_count == 1 and unfavorable_count >= 2:
            overall = "Concerning - More unfavorable patterns than favorable"
        else:
            overall = "Concerning - Few favorable patterns"

        return {
            "overall_association": overall,
            "pattern_summary": {
                "favorable": favorable_count,
                "unfavorable": unfavorable_count,
                "unknown": unknown_count
            },
            "patterns": patterns,
            "concerns": self._generate_concerns(stress, sleep, activity, favorable_count, unfavorable_count)
        }

    def _generate_concerns(self, stress: dict, sleep: dict, activity: dict, favorable: int, unfavorable: int) -> list:
        """Generate list of main concerns based on patterns"""
        concerns = []

        if stress.get("favorability") == "unfavorable":
            concerns.append("High stress levels may impair glucose control")

        if sleep.get("favorability") == "unfavorable":
            avg_sleep = sleep.get("avg_sleep_hours")
            if avg_sleep and avg_sleep < 7:
                concerns.append("Insufficient sleep (<7 hours) can worsen glucose patterns")
            elif avg_sleep and avg_sleep > 9:
                concerns.append("Excessive sleep (>9 hours) may indicate sleep issues affecting glucose")

        if activity.get("favorability") == "unfavorable":
            concerns.append("Low activity levels (<4,000 steps) reduce glucose utilization")

        if unfavorable >= 2:
            concerns.append("Multiple risk factors present - comprehensive lifestyle intervention recommended")

        if not concerns:
            concerns.append("No significant concerns identified - current lifestyle patterns support glucose control")

        return concerns

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
             from_date: Optional[str] = None, to_date: Optional[str] = None,
             period: Optional[str] = None) -> str:
        """Run correlation analysis"""

        patient_id, resolution_error = self._resolve_patient_id(patient_id, patient_name)
        if resolution_error:
            return resolution_error

        # Import trend tools to get data
        try:
            from tools.health_progress_tool import StressHRVTrendTool, SleepTrendTool, ActivityTrendTool

            stress_tool = StressHRVTrendTool()
            sleep_tool = SleepTrendTool()
            activity_tool = ActivityTrendTool()

            # Set user context
            user_context = getattr(self, 'user_context', None)
            if user_context:
                stress_tool.set_user_context(user_context)
                sleep_tool.set_user_context(user_context)
                activity_tool.set_user_context(user_context)

            # Fetch data from each tool
            stress_result = stress_tool._run(patient_id=patient_id, from_date=from_date, to_date=to_date, period=period)
            sleep_result = sleep_tool._run(patient_id=patient_id, from_date=from_date, to_date=to_date, period=period)
            activity_result = activity_tool._run(patient_id=patient_id, from_date=from_date, to_date=to_date, period=period)

            # Parse results
            stress_data = json.loads(stress_result).get("summary", {}) if stress_result else {}
            sleep_data = json.loads(sleep_result).get("summary", {}) if sleep_result else {}
            activity_data = json.loads(activity_result).get("summary", {}) if activity_result else {}

        except Exception as e:
            logger.error(f"Error fetching correlation data: {e}")
            return json.dumps({
                "error": f"Failed to analyze correlations: {str(e)}",
                "patient_id": patient_id
            })

        # Analyze patterns
        stress_analysis = self._analyze_stress_pattern(stress_data)
        sleep_analysis = self._analyze_sleep_pattern(sleep_data)
        activity_analysis = self._analyze_activity_pattern(activity_data)

        # Calculate overall association
        overall = self._calculate_overall_association(stress_analysis, sleep_analysis, activity_analysis)

        # Build response
        response = {
            "patient_id": patient_id,
            "analysis": {
                "stress": stress_analysis,
                "sleep": sleep_analysis,
                "activity": activity_analysis
            },
            "overall_association": overall["overall_association"],
            "pattern_count": overall["pattern_summary"],
            "patterns": overall["patterns"],
            "concerns": overall["concerns"],
            "message": f"Correlation analysis complete. {overall['overall_association']}"
        }

        return json.dumps(response)

    async def _arun(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
                   from_date: Optional[str] = None, to_date: Optional[str] = None,
                   period: Optional[str] = None) -> str:
        return self._run(patient_id, patient_name, from_date, to_date, period)