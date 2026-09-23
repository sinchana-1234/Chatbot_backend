#!/usr/bin/env python3
"""
Correlation Analysis Tool for Revival Medical System

Analyzes how stress, sleep, and activity CORRELATE WITH patient's glucose patterns.
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

    Returns: Correlation analysis showing:
    - How stress levels correlate with glucose (positive/negative/none)
    - How sleep duration correlates with glucose (positive/negative/none)
    - How activity levels correlate with glucose (positive/negative/none)
    - Overall pattern: which factor has strongest impact on glucose
    - Actionable insights based on correlations

    Use for questions like:
    - "How does activity affect this patient's glucose?"
    - "How do sleep and stress relate to glucose?"
    - "What lifestyle factor impacts glucose the most?"
    - "Are there patterns between activity and glucose spikes?"
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

    def _calculate_correlation(self, factor_values: list, glucose_values: list) -> dict:
        """
        Calculate simple correlation between a lifestyle factor and glucose
        Returns: correlation strength (positive/negative/none) and strength value
        """
        if len(factor_values) < 2 or len(glucose_values) < 2:
            return {
                "status": "insufficient_data",
                "correlation": "unknown",
                "strength": None,
                "message": "Not enough data points to calculate correlation"
            }

        # Simple correlation: if factor goes up, does glucose go up too?
        factor_avg = sum(factor_values) / len(factor_values)
        glucose_avg = sum(glucose_values) / len(glucose_values)

        numerator = sum((f - factor_avg) * (g - glucose_avg) for f, g in zip(factor_values, glucose_values))
        factor_variance = sum((f - factor_avg) ** 2 for f in factor_values)
        glucose_variance = sum((g - glucose_avg) ** 2 for g in glucose_values)

        if factor_variance == 0 or glucose_variance == 0:
            return {
                "status": "no_variance",
                "correlation": "none",
                "strength": 0,
                "message": "No variance in data to correlate"
            }

        correlation = numerator / (factor_variance * glucose_variance) ** 0.5

        if correlation > 0.3:
            corr_type = "positive"
            interpretation = "higher factor values tend to associate with higher glucose"
        elif correlation < -0.3:
            corr_type = "negative"
            interpretation = "higher factor values tend to associate with lower glucose"
        else:
            corr_type = "weak/none"
            interpretation = "little to no clear relationship"

        return {
            "status": "ok",
            "correlation": corr_type,
            "strength": round(correlation, 2),
            "interpretation": interpretation,
            "data_points": len(factor_values)
        }

    def _get_glucose_data(self, patient_id: int, from_date: Optional[str] = None, to_date: Optional[str] = None) -> list:
        """Fetch glucose readings for the patient"""
        try:
            with DatabaseManager() as db_manager:
                query = "SELECT reading_value, reading_date FROM patient_glucose_readings WHERE patient_id = %s"
                params = [patient_id]

                if from_date:
                    query += " AND reading_date >= %s"
                    params.append(from_date)
                if to_date:
                    query += " AND reading_date <= %s"
                    params.append(to_date)

                query += " ORDER BY reading_date"

                results = db_manager.execute_query(query, params)
                return results if results else []
        except Exception as e:
            logger.error(f"Error fetching glucose data: {e}")
            return []

    def _get_stress_data(self, patient_id: int, from_date: Optional[str] = None, to_date: Optional[str] = None) -> list:
        """Fetch stress readings for the patient"""
        try:
            with DatabaseManager() as db_manager:
                query = "SELECT stress_percentage, reading_date FROM patient_stress_hrv WHERE patient_id = %s"
                params = [patient_id]

                if from_date:
                    query += " AND reading_date >= %s"
                    params.append(from_date)
                if to_date:
                    query += " AND reading_date <= %s"
                    params.append(to_date)

                query += " ORDER BY reading_date"

                results = db_manager.execute_query(query, params)
                return results if results else []
        except Exception as e:
            logger.error(f"Error fetching stress data: {e}")
            return []

    def _get_sleep_data(self, patient_id: int, from_date: Optional[str] = None, to_date: Optional[str] = None) -> list:
        """Fetch sleep duration for the patient"""
        try:
            with DatabaseManager() as db_manager:
                query = "SELECT (deep_sleep_mins + light_sleep_mins) / 60.0 as sleep_hours, reading_date FROM patient_sleep WHERE patient_id = %s"
                params = [patient_id]

                if from_date:
                    query += " AND reading_date >= %s"
                    params.append(from_date)
                if to_date:
                    query += " AND reading_date <= %s"
                    params.append(to_date)

                query += " ORDER BY reading_date"

                results = db_manager.execute_query(query, params)
                return results if results else []
        except Exception as e:
            logger.error(f"Error fetching sleep data: {e}")
            return []

    def _get_activity_data(self, patient_id: int, from_date: Optional[str] = None, to_date: Optional[str] = None) -> list:
        """Fetch activity (steps) for the patient"""
        try:
            with DatabaseManager() as db_manager:
                query = "SELECT steps, activity_date FROM patient_activity WHERE patient_id = %s"
                params = [patient_id]

                if from_date:
                    query += " AND activity_date >= %s"
                    params.append(from_date)
                if to_date:
                    query += " AND activity_date <= %s"
                    params.append(to_date)

                query += " ORDER BY activity_date"

                results = db_manager.execute_query(query, params)
                return results if results else []
        except Exception as e:
            logger.error(f"Error fetching activity data: {e}")
            return []

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
             from_date: Optional[str] = None, to_date: Optional[str] = None,
             period: Optional[str] = None) -> str:
        """Run correlation analysis"""

        patient_id, resolution_error = self._resolve_patient_id(patient_id, patient_name)
        if resolution_error:
            return resolution_error

        try:
            # Fetch all data
            glucose_data = self._get_glucose_data(patient_id, from_date, to_date)
            stress_data = self._get_stress_data(patient_id, from_date, to_date)
            sleep_data = self._get_sleep_data(patient_id, from_date, to_date)
            activity_data = self._get_activity_data(patient_id, from_date, to_date)

            if not glucose_data:
                return json.dumps({
                    "error": "No glucose data found for this patient in the specified period",
                    "patient_id": patient_id
                })

            # Align dates for correlation (same dates across all datasets)
            glucose_dict = {str(g[1]): g[0] for g in glucose_data}
            stress_dict = {str(s[1]): s[0] for s in stress_data}
            sleep_dict = {str(s[1]): s[0] for s in sleep_data}
            activity_dict = {str(a[1]): a[0] for a in activity_data}

            # Find common dates
            common_dates = set(glucose_dict.keys())
            if stress_data:
                common_dates &= set(stress_dict.keys())
            if sleep_data:
                common_dates &= set(sleep_dict.keys())
            if activity_data:
                common_dates &= set(activity_dict.keys())

            common_dates = sorted(list(common_dates))

            if not common_dates:
                return json.dumps({
                    "error": "No overlapping dates found between glucose and lifestyle data",
                    "patient_id": patient_id,
                    "message": "Need data on the same dates to calculate correlations"
                })

            # Extract aligned values
            glucose_values = [glucose_dict[d] for d in common_dates]
            stress_values = [stress_dict.get(d, None) for d in common_dates]
            sleep_values = [sleep_dict.get(d, None) for d in common_dates]
            activity_values = [activity_dict.get(d, None) for d in common_dates]

            # Calculate correlations
            stress_correlation = self._calculate_correlation(
                [s for s in stress_values if s is not None],
                [glucose_dict[common_dates[i]] for i, s in enumerate(stress_values) if s is not None]
            ) if any(s is not None for s in stress_values) else {"status": "no_data", "correlation": "unknown"}

            sleep_correlation = self._calculate_correlation(
                [s for s in sleep_values if s is not None],
                [glucose_dict[common_dates[i]] for i, s in enumerate(sleep_values) if s is not None]
            ) if any(s is not None for s in sleep_values) else {"status": "no_data", "correlation": "unknown"}

            activity_correlation = self._calculate_correlation(
                [a for a in activity_values if a is not None],
                [glucose_dict[common_dates[i]] for i, a in enumerate(activity_values) if a is not None]
            ) if any(a is not None for a in activity_values) else {"status": "no_data", "correlation": "unknown"}

            # Build response
            response = {
                "patient_id": patient_id,
                "period_analyzed": f"{common_dates[0]} to {common_dates[-1]}" if common_dates else "N/A",
                "data_points_aligned": len(common_dates),
                "correlations": {
                    "stress_vs_glucose": stress_correlation,
                    "sleep_vs_glucose": sleep_correlation,
                    "activity_vs_glucose": activity_correlation
                },
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

    def _generate_summary(self, stress_corr: dict, sleep_corr: dict, activity_corr: dict) -> str:
        """Generate human-readable summary of correlations"""
        findings = []

        if stress_corr.get("status") == "ok":
            if stress_corr["correlation"] == "positive":
                findings.append(f"Higher stress is associated with higher glucose levels ({stress_corr['strength']})")
            elif stress_corr["correlation"] == "negative":
                findings.append(f"Higher stress is associated with lower glucose levels ({stress_corr['strength']})")

        if sleep_corr.get("status") == "ok":
            if sleep_corr["correlation"] == "positive":
                findings.append(f"More sleep is associated with higher glucose levels ({sleep_corr['strength']})")
            elif sleep_corr["correlation"] == "negative":
                findings.append(f"Less sleep is associated with higher glucose levels ({sleep_corr['strength']})")

        if activity_corr.get("status") == "ok":
            if activity_corr["correlation"] == "positive":
                findings.append(f"More activity is associated with higher glucose levels ({activity_corr['strength']})")
            elif activity_corr["correlation"] == "negative":
                findings.append(f"More activity is associated with lower glucose levels ({activity_corr['strength']})")

        if not findings:
            return "Insufficient data to identify clear correlations with glucose"

        return " | ".join(findings)

    async def _arun(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
                   from_date: Optional[str] = None, to_date: Optional[str] = None,
                   period: Optional[str] = None) -> str:
        return self._run(patient_id, patient_name, from_date, to_date, period)