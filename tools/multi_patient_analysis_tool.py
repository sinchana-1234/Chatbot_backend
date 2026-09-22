#!/usr/bin/env python3
"""
Multi-Patient Analysis Tool (PostgreSQL version)
Finds distinct patients with high/low readings, scoped to the requesting
doctor's own patients by default — mirrors the pattern in
SpecificMedicalValueTool (same Postgres tables/columns, same doctor-scoping).
"""

import logging
import json
from typing import Optional, ClassVar, Dict, Tuple
from datetime import datetime
from langchain.tools import BaseTool
from sqlalchemy import text

from dal.postgres_db import SessionLocalPG

logger = logging.getLogger(__name__)


class MultiPatientAnalysisTool(BaseTool):
    """Find distinct patients with high/low readings, using live PostgreSQL data."""
    name: str = "analyze_multiple_patients"
    description: str = """Analyze medical readings across multiple patients to find distinct
    patients with high/low values, scoped to the requesting doctor's own patients.

    Parameters:
    - reading_type (str): "glucose", "blood_pressure", "body_temperature", "hrv", "spo2", "stress"
    - date_filter (str): Date in YYYY-MM-DD format (OPTIONAL — if not provided, analyzes all available data)
    - analysis_type (str): "high" or "low" to find patients with concerning values
    - custom_threshold (float): OPTIONAL — a specific numeric threshold if the user gives one
        (e.g. "more than 200" → custom_threshold=200). If not given, uses the standard clinical
        threshold for that reading type.

    Returns a list of DISTINCT patients (from the requesting doctor's own patient list) who
    have readings above/below a clinical threshold, with each patient's highest/lowest value
    and sample readings.

    Use this for queries like:
    - "List patients with high glucose readings" (no date needed)
    - "Find patients with low blood pressure"
    - "List all patients whose sugar value is high on a specific date"
    """

    def set_user_context(self, user_context):
        object.__setattr__(self, 'user_context', user_context)

    # Same table/column mapping as SpecificMedicalValueTool — kept identical
    # on purpose so both tools agree about where each reading type lives.
    TABLE_MAP: ClassVar[Dict[str, Tuple[str, str, str]]] = {
        "glucose": ("glucose_readings", "glucose_value", "local_event_time"),
        "blood_pressure": ("blood_pressure_readings", "systolic", "actual_time"),
        "body_temperature": ("body_temperature_readings", "temperature", "actual_time"),
        "hrv": ("hrv_readings", "value", "actual_time"),
        "spo2": ("spo2_readings", "value", "actual_time"),
        "stress": ("stress_readings", "value", "actual_time"),
    }

    # Same clinical thresholds as the old MySQL-based medical_readings_service,
    # preserved as-is — this fix changes the DATA SOURCE, not the thresholds.
    THRESHOLDS: ClassVar[Dict[str, Dict[str, float]]] = {
        "glucose": {"high": 180, "low": 70},
        "blood_pressure": {"high": 140, "low": 90},
        "body_temperature": {"high": 100.4, "low": 96.0},
        "hrv": {"high": 50, "low": 20},
        "spo2": {"high": 100, "low": 90},
        "stress": {"high": 80, "low": 20},
    }

    def _run(self, reading_type: str = "glucose", date_filter: Optional[str] = None,
              analysis_type: str = "high", custom_threshold: Optional[float] = None) -> str:
        try:
            if reading_type not in self.TABLE_MAP:
                return json.dumps({
                    "error": f"Invalid reading type: {reading_type}. "
                             f"Available types: {list(self.TABLE_MAP.keys())}"
                })
            if analysis_type not in ("high", "low"):
                return json.dumps({"error": "analysis_type must be 'high' or 'low'"})

            table, column, time_col = self.TABLE_MAP[reading_type]
            threshold = custom_threshold if custom_threshold is not None else self.THRESHOLDS[reading_type][analysis_type]

            user_context = getattr(self, 'user_context', None)

            # --- Doctor-scoping (the actual fix — was previously a dead
            # `all_patients` parameter that did nothing) ---
            # Staff see only their OWN assigned patients by default, matching
            # every other patient-facing tool in this codebase. Patients get
            # forced to themselves (a "find high readings across patients"
            # question doesn't make sense for a patient role, but we don't
            # want to error — just scope it to their own single record).
            from dal.database import DatabaseManager
            with DatabaseManager() as db_manager:
                if user_context and user_context.get('role_id') == 1:
                    patient_ids = [user_context.get('user_id')]
                    id_to_name = {user_context.get('user_id'): "You"}
                else:
                    doctor_id = user_context.get('user_id') if user_context else None
                    own_patients = db_manager.get_doctor_patients(doctor_user_id=doctor_id) if doctor_id else []
                    patient_ids = [p["patient_id"] for p in own_patients]
                    id_to_name = {
                        p["patient_id"]: f"{p.get('patient_first_name') or ''} {p.get('patient_last_name') or ''}".strip()
                        for p in own_patients
                    }

            if not patient_ids:
                return json.dumps({
                    "message": "No patients are currently assigned to you.",
                    "distinct_patients": []
                })

            # --- Date filter ---
            params = {"threshold": threshold}
            if date_filter:
                try:
                    datetime.strptime(date_filter, "%Y-%m-%d")
                except ValueError:
                    return json.dumps({"error": "Invalid date format. Use YYYY-MM-DD"})
                date_condition = f"DATE({time_col}) = :date"
                params["date"] = date_filter
            else:
                date_condition = "1=1"

            comparator = ">" if analysis_type == "high" else "<"

            # patient_id IN (...) built as bound params, not string-interpolated,
            # to avoid any SQL injection risk from a large/odd patient_ids list.
            id_params = {f"pid{i}": pid for i, pid in enumerate(patient_ids)}
            id_placeholder = ", ".join(f":{k}" for k in id_params)
            params.update(id_params)

            db = SessionLocalPG()
            try:
                query = f"""
                    SELECT patient_id, {column}, {time_col}
                    FROM {table}
                    WHERE patient_id IN ({id_placeholder})
                      AND {column} {comparator} :threshold
                      AND {date_condition}
                    ORDER BY patient_id, {time_col} DESC
                """
                results = db.execute(text(query), params).fetchall()
            finally:
                db.close()

            if not results:
                return json.dumps({
                    "reading_type": reading_type,
                    "analysis_type": analysis_type,
                    "threshold": threshold,
                    "date_filter": date_filter or "All dates",
                    "distinct_patients": [],
                    "message": f"No patients found with {analysis_type} {reading_type} readings."
                })

            # Group by patient, keep up to 3 sample readings each
            by_patient = {}
            for pid, value, reading_time in results:
                by_patient.setdefault(pid, []).append({
                    "value": float(value),
                    "time": str(reading_time)
                })

            distinct_patients = []
            for pid, readings in by_patient.items():
                values = [r["value"] for r in readings]
                extreme_value = max(values) if analysis_type == "high" else min(values)
                distinct_patients.append({
                    "patient_id": pid,
                    "patient_name": id_to_name.get(pid, f"Patient {pid}"),
                    "extreme_value": extreme_value,
                    "sample_readings": readings[:3],
                    "total_readings": len(readings)
                })

            return json.dumps({
                "reading_type": reading_type,
                "analysis_type": analysis_type,
                "threshold": threshold,
                "date_filter": date_filter or "All dates",
                "distinct_patients": distinct_patients,
                "total_patients": len(distinct_patients),
                "message": f"Found {len(distinct_patients)} distinct patients with "
                           f"{analysis_type} {reading_type} readings."
            }, indent=2)

        except Exception as e:
            logger.error(f"Error in MultiPatientAnalysisTool: {e}")
            return json.dumps({"error": f"Error analyzing multiple patients: {str(e)}"})

    async def _arun(self, reading_type="glucose", date_filter=None, analysis_type="high", custom_threshold=None):
        return self._run(reading_type, date_filter, analysis_type, custom_threshold)