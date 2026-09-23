#!/usr/bin/env python3
"""
Correlation Analysis Tool for Revival Medical System

Analyzes how stress, sleep, and activity correlate WITH patient's glucose patterns.
Uses _fetch() with proper auth token, cycle window, and date resolution.
"""

import json
import logging
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from langchain.tools import BaseTool
from dal.database import DatabaseManager

logger = logging.getLogger(__name__)


class CorrelationAnalysisTool(BaseTool):
    """Analyzes correlations between lifestyle factors and glucose"""

    name: str = "analyze_glucose_correlations"
    description: str = """Analyze how stress, sleep, and activity correlate with a patient's glucose patterns.

    Parameters:
    - patient_id (int): Patient ID (optional for patient role, required for staff)
    - patient_name (str): Patient name (alternative to patient_id)
    - from_date (str): Start date (YYYY-MM-DD) — OPTIONAL
    - to_date (str): End date (YYYY-MM-DD) — OPTIONAL
    - period (str): OPTIONAL relative phrase, e.g. 'last 30 days', 'this month'

    Returns: Correlation analysis showing:
    - How stress levels correlate with glucose (positive/negative/weak)
    - How sleep duration correlates with glucose (positive/negative/weak)
    - How activity levels correlate with glucose (positive/negative/weak)
    - Which factor has strongest impact on glucose
    - Actionable insights based on data patterns

    Use for questions like:
    - "How does activity affect this patient's glucose?"
    - "How do sleep and stress relate to glucose?"
    - "What lifestyle factor impacts glucose the most?"
    """

    def __init__(self):
        super().__init__()

    def set_user_context(self, user_context):
        object.__setattr__(self, 'user_context', user_context)

    def _resolve_patient_id(self, patient_id: Optional[int], patient_name: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
        """Resolve patient name to ID if needed"""
        if patient_id:
            return patient_id, None
        if not patient_name:
            return None, json.dumps({
                "status": "not_found",
                "notice": "Either patient_id or patient_name is required."
            })

        user_context = getattr(self, 'user_context', None)

        try:
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
                                {"id": p["patient_id"], "name": f"{p.get('patient_first_name') or ''} {p.get('patient_last_name') or ''}".strip()}
                                for p in own_matching
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
                    return None, json.dumps({"status": "not_found", "notice": f"No patient found with name containing '{patient_name}'."})

                if len(matching_users) > 1:
                    return None, json.dumps({
                        "status": "ambiguous_name",
                        "notice": f"Multiple patients match '{patient_name}'.",
                        "matching_patients": [
                            {"id": u.id, "name": f"{u.first_name or ''} {u.last_name or ''}".strip()}
                            for u in matching_users
                        ]
                    })

                return matching_users[0].id, None
        except Exception as e:
            logger.error(f"Error resolving patient: {e}")
            return None, json.dumps({"error": f"Failed to resolve patient: {str(e)}"})

    def _fetch_all_data(self, patient_id: int, from_date: Optional[str] = None,
                        to_date: Optional[str] = None, period: Optional[str] = None) -> Dict:
        """Fetch all per-day data via the trend tool's _fetch, with the SAME
        auth-token and cycle-window resolution _run_for_metric uses."""
        try:
            from tools.health_progress_tool import GlucoseTrendTool
            from dal.postgres_db import resolve_range
            from config import settings

            tool = GlucoseTrendTool()
            user_context = getattr(self, 'user_context', None)
            if user_context and hasattr(tool, 'set_user_context'):
                tool.set_user_context(user_context)

            # 1. Auth token — same source as _run_for_metric (line 587)
            auth_token = (user_context.get('auth_token') or user_context.get('token')) if user_context else None
            if not auth_token:
                logger.error("No auth token found in user context")
                return {}

            # 2. Date window — explicit dates win; else current cycle; else wide floor
            start, end, _ = resolve_range(from_date, to_date, period)
            if not start:
                cyc = tool._cycle_window(patient_id)
                if cyc:
                    start, end = cyc
                else:
                    end = datetime.now().strftime("%Y-%m-%d")
                    start = (datetime.now()
                             - timedelta(days=settings.TREND_ALL_HISTORY_LOOKBACK_DAYS)
                             ).strftime("%Y-%m-%d")

            object.__setattr__(self, '_resolved_window', (start, end))
            logger.debug(f"Fetching correlation data for patient {patient_id} from {start} to {end}")
            return tool._fetch(patient_id, start, end, auth_token)
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return {}

    def _extract_daily_series(self, data: Dict) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Extract per-day series from raw fetch data
        Returns: (daily_glucose, daily_stress, daily_sleep, daily_activity)
        """
        daily_glucose = {}
        daily_stress = {}
        daily_sleep = {}
        daily_activity = {}

        try:
            # Extract glucose: glucoseDailyAnalytics has glucoseDate and meanGlucose
            if "glucoseDailyAnalytics" in data and isinstance(data["glucoseDailyAnalytics"], list):
                for day in data["glucoseDailyAnalytics"]:
                    date_key = day.get("glucoseDate", "")[:10]  # YYYY-MM-DD
                    glucose_val = day.get("meanGlucose")
                    if date_key and glucose_val is not None:
                        daily_glucose[date_key] = float(glucose_val)

            # Extract stress: stress list has readingDate and average
            if "stress" in data and isinstance(data["stress"], list):
                for day in data["stress"]:
                    date_key = day.get("readingDate", "")[:10]  # YYYY-MM-DD
                    stress_val = day.get("average")
                    if date_key and stress_val is not None:
                        daily_stress[date_key] = float(stress_val)

            # Extract sleep: sleep list has readingDate and sleep stage minutes
            if "sleep" in data and isinstance(data["sleep"], list):
                for day in data["sleep"]:
                    date_key = day.get("readingDate", "")[:10]  # YYYY-MM-DD
                    deep = day.get("deepSleep", 0) or 0
                    light = day.get("lightSleep", 0) or 0
                    rem = day.get("remSleep", 0) or 0
                    total_mins = deep + light + rem
                    if date_key and total_mins > 0:
                        daily_sleep[date_key] = total_mins / 60.0  # Convert to hours

            # Extract activity: activity list has readingDate and totalSteps
            if "activity" in data and isinstance(data["activity"], list):
                for day in data["activity"]:
                    date_key = day.get("readingDate", "")[:10]  # YYYY-MM-DD
                    steps = day.get("totalSteps")
                    if date_key and steps is not None:
                        daily_activity[date_key] = float(steps)

        except Exception as e:
            logger.error(f"Error extracting daily series: {e}")

        return daily_glucose, daily_stress, daily_sleep, daily_activity

    def _calculate_correlation(self, factor_values: List[float], glucose_values: List[float]) -> Dict:
        """
        Calculate Pearson correlation between factor and glucose
        Minimum 10 paired days required for medical significance
        """
        if len(factor_values) < 5:
            return {
                "status": "insufficient_data",
                "correlation": "unknown",
                "strength": None,
                "message": f"Only {len(factor_values)} paired days. Need at least 5 to estimate a correlation."
            }

        if len(factor_values) != len(glucose_values):
            return {
                "status": "error",
                "correlation": "unknown",
                "strength": None,
                "message": "Mismatched data lengths"
            }

        # Calculate Pearson correlation coefficient
        factor_mean = sum(factor_values) / len(factor_values)
        glucose_mean = sum(glucose_values) / len(glucose_values)

        numerator = sum((f - factor_mean) * (g - glucose_mean) for f, g in zip(factor_values, glucose_values))
        factor_variance = sum((f - factor_mean) ** 2 for f in factor_values)
        glucose_variance = sum((g - glucose_mean) ** 2 for g in glucose_values)

        if factor_variance == 0 or glucose_variance == 0:
            return {
                "status": "no_variance",
                "correlation": "none",
                "strength": 0,
                "message": "No variance in data to correlate"
            }

        correlation = numerator / (factor_variance * glucose_variance) ** 0.5
        correlation = round(correlation, 2)

        # Classify correlation
        if correlation > 0.4:
            corr_type = "positive"
            strength_desc = "strong positive"
            interpretation = "higher values tend to associate with higher glucose"
        elif correlation > 0.2:
            corr_type = "positive"
            strength_desc = "moderate positive"
            interpretation = "higher values show some association with higher glucose"
        elif correlation < -0.4:
            corr_type = "negative"
            strength_desc = "strong negative"
            interpretation = "higher values tend to associate with lower glucose"
        elif correlation < -0.2:
            corr_type = "negative"
            strength_desc = "moderate negative"
            interpretation = "higher values show some association with lower glucose"
        else:
            corr_type = "weak"
            strength_desc = "weak"
            interpretation = "little to no clear relationship"

        reliability = "preliminary" if len(factor_values) < 10 else "reliable"
        return {
            "status": "ok",
            "correlation": corr_type,
            "strength": correlation,
            "strength_desc": strength_desc,
            "interpretation": interpretation,
            "data_points": len(factor_values),
            "reliability": reliability,
            "reliability_note": (
                f"Based on {len(factor_values)} paired days — treat as a preliminary signal, "
                "not a confirmed relationship."
            ) if len(factor_values) < 10 else None
        }

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
             from_date: Optional[str] = None, to_date: Optional[str] = None,
             period: Optional[str] = None) -> str:
        """Run correlation analysis"""

        patient_id, resolution_error = self._resolve_patient_id(patient_id, patient_name)
        if resolution_error:
            return resolution_error

        try:
            # Fetch all per-day data in one call with proper auth and cycle window
            data = self._fetch_all_data(patient_id, from_date, to_date, period)

            if not data:
                return json.dumps({
                    "error": "Failed to fetch data for this patient",
                    "patient_id": patient_id
                })

            # Extract daily series
            daily_glucose, daily_stress, daily_sleep, daily_activity = self._extract_daily_series(data)

            if not daily_glucose:
                return json.dumps({
                    "error": "No glucose data found for this patient in the specified period",
                    "patient_id": patient_id
                })

            # Calculate correlations independently (each factor intersects with glucose separately)
            stress_correlation = self._correlate_independently(daily_stress, daily_glucose, "stress")
            sleep_correlation = self._correlate_independently(daily_sleep, daily_glucose, "sleep")
            activity_correlation = self._correlate_independently(daily_activity, daily_glucose, "activity")

            # Find strongest correlation
            correlations = [
                ("stress", stress_correlation),
                ("sleep", sleep_correlation),
                ("activity", activity_correlation)
            ]
            valid_correlations = [(name, corr) for name, corr in correlations if corr.get("status") == "ok"]
            strongest = max(valid_correlations, key=lambda x: abs(x[1]["strength"]))[0] if valid_correlations else None

            # Build response
            response = {
                "patient_id": patient_id,
                "period": f"{self._resolved_window[0]} to {self._resolved_window[1]}"
                          if getattr(self, '_resolved_window', None) else "unknown",
                "correlations": {
                    "stress_vs_glucose": stress_correlation,
                    "sleep_vs_glucose": sleep_correlation,
                    "activity_vs_glucose": activity_correlation
                },
                "strongest_factor": strongest,
                "summary": self._generate_summary(stress_correlation, sleep_correlation, activity_correlation),
                "message": "Correlation analysis complete"
            }

            return json.dumps(response)

        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return json.dumps({
                "error": f"Failed to analyze correlations: {str(e)}",
                "patient_id": patient_id
            })

    def _correlate_independently(self, factor_data: Dict[str, float], glucose_data: Dict[str, float], factor_name: str) -> Dict:
        """Correlate one factor with glucose on their shared dates"""
        if not factor_data:
            return {
                "status": "no_data",
                "correlation": "unknown",
                "message": f"No {factor_name} data available"
            }

        # Find common dates between this factor and glucose
        common_dates = set(factor_data.keys()) & set(glucose_data.keys())

        if not common_dates:
            return {
                "status": "no_overlap",
                "correlation": "unknown",
                "message": f"No overlapping dates between {factor_name} and glucose"
            }

        # Extract paired values
        factor_values = [factor_data[d] for d in sorted(common_dates)]
        glucose_values = [glucose_data[d] for d in sorted(common_dates)]

        return self._calculate_correlation(factor_values, glucose_values)

    def _generate_summary(self, stress_corr: Dict, sleep_corr: Dict, activity_corr: Dict) -> str:
        """Generate human-readable summary"""
        findings = []

        if stress_corr.get("status") == "ok":
            findings.append(f"Stress: {stress_corr['strength_desc']} correlation ({stress_corr['strength']})")

        if sleep_corr.get("status") == "ok":
            findings.append(f"Sleep: {sleep_corr['strength_desc']} correlation ({sleep_corr['strength']})")

        if activity_corr.get("status") == "ok":
            findings.append(f"Activity: {activity_corr['strength_desc']} correlation ({activity_corr['strength']})")

        if not findings:
            return "Insufficient overlapping data to identify correlations"

        return " | ".join(findings)

    async def _arun(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
                   from_date: Optional[str] = None, to_date: Optional[str] = None,
                   period: Optional[str] = None) -> str:
        return self._run(patient_id, patient_name, from_date, to_date, period)