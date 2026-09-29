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
from datetime import datetime, timedelta
from langchain.tools import BaseTool
from sqlalchemy import text

from dal.postgres_db import SessionLocalPG

logger = logging.getLogger(__name__)

def _dmy(iso):
    """'2026-09-10' -> '10-09-2026'."""
    parts = str(iso).split("-")
    return "-".join(reversed(parts)) if len(parts) == 3 else str(iso)

_UNITS = {
    "glucose": "mg/dL", "blood_pressure": "mmHg", "body_temperature": "°F",
    "hrv": "ms", "spo2": "%", "stress": "/100",
}

def _fmt_ts(ts):
    """datetime or 'YYYY-MM-DD HH:MM:SS' -> 'DD-MM-YYYY HH:MM'."""
    if hasattr(ts, "strftime"):
        return ts.strftime("%d-%m-%Y %H:%M")
    s = str(ts)
    try:
        d, t = s.split(" ")[0], s.split(" ")[1][:5]
        return "-".join(reversed(d.split("-"))) + " " + t
    except Exception:
        return s

class MultiPatientAnalysisTool(BaseTool):
    """Find distinct patients with high/low readings, using live PostgreSQL data."""
    name: str = "analyze_multiple_patients"
    description: str = """Analyze medical readings across multiple patients to find distinct
    patients with high/low values, scoped to the requesting doctor's own patients.

    Parameters:
    - reading_type (str): "glucose", "blood_pressure", "body_temperature", "hrv", "spo2", "stress"
    - from_date / to_date (str): OPTIONAL date range in YYYY-MM-DD. When BOTH are given, only
        readings in that range are considered.
    - date_filter (str): OPTIONAL single date in YYYY-MM-DD (one day). Ignored if from_date/to_date given.
    - analysis_type (str): "high" or "low" to find patients with concerning values
    - custom_threshold (float): OPTIONAL — a specific numeric threshold if the user gives one
        (e.g. "more than 200" → custom_threshold=200). If not given, uses the standard clinical
        threshold for that reading type.

    DEFAULT WINDOW: if NO date is given at all, only the LAST 24 HOURS are considered (not all
    history) — this answers "who is currently high/low".

    Returns a finished, ready-to-send summary (relay it verbatim, do not reformat).

    Use this ONLY for cross-roster questions that scan MANY patients — the query
    asks "which patients", "list patients", "who has", "how many patients", with
    NO single patient named:
    - "Which patients have high glucose readings" (defaults to last 24 hours)
    - "Who had low blood pressure"
    - "Which patients had high sugar between 2026-09-10 and 2026-09-20" (from_date/to_date)
    - "List all patients whose sugar is high on a specific date" (date_filter)

    DO NOT use this tool when the query names ONE specific patient (e.g.
    "has water two had any lows", "is Vicky's sugar high", "does <name> go low
    at night"). A named-patient question is a SINGLE-patient question — route it
    to get_specific_medical_value (analysis_type=pattern_high / pattern_low),
    which filters to that one patient. This tool cannot filter by patient name
    and will scan the whole roster instead, returning the WRONG patient.
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
        "glucose": {"high": 200, "low": 70},
        "blood_pressure": {"high": 140, "low": 90},
        "body_temperature": {"high": 100.4, "low": 96.0},
        "hrv": {"high": 50, "low": 20},
        "spo2": {"high": 100, "low": 90},
        "stress": {"high": 80, "low": 20},
    }

    def _run(self, reading_type: str = "glucose", date_filter: Optional[str] = None,
              analysis_type: str = "high", custom_threshold: Optional[float] = None,
              from_date: Optional[str] = None, to_date: Optional[str] = None) -> str:
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
                answer = "No patients are currently assigned to you."
                if user_context is not None:
                    user_context["_last_multi_patient_text"] = answer
                return answer

            # --- Date filter ---
            params = {"threshold": threshold}
            try:
                for d in (from_date, to_date, date_filter):
                    if d:
                        datetime.strptime(d, "%Y-%m-%d")
            except ValueError:
                return json.dumps({"error": "Invalid date format. Use YYYY-MM-DD"})

            if from_date and to_date:
                date_condition = f"DATE({time_col}) BETWEEN :dfrom AND :dto"
                params["dfrom"], params["dto"] = from_date, to_date
                period_desc = f"from {_dmy(from_date)} to {_dmy(to_date)}"
            elif date_filter:
                date_condition = f"DATE({time_col}) = :date"
                params["date"] = date_filter
                period_desc = f"on {_dmy(date_filter)}"
            else:
                params["since"] = datetime.now() - timedelta(hours=24)
                date_condition = f"{time_col} >= :since"
                period_desc = "in the last 24 hours"

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
                answer = f"No patients had {analysis_type} {reading_type} readings {period_desc}."
                if user_context is not None:
                    user_context["_last_multi_patient_text"] = answer
                return answer

            # Group by patient; keep each reading's value + time so we can name
            # the extreme reading and format its timestamp deterministically.
            by_patient = {}
            for pid, value, reading_time in results:
                by_patient.setdefault(pid, []).append((float(value), reading_time))

            sup = "highest" if analysis_type == "high" else "lowest"
            unit = _UNITS.get(reading_type, "")
            lines = []
            for pid, readings in by_patient.items():
                if analysis_type == "high":
                    ext_val, ext_time = max(readings, key=lambda r: r[0])
                else:
                    ext_val, ext_time = min(readings, key=lambda r: r[0])
                name = id_to_name.get(pid, f"Patient {pid}")
                n = len(readings)
                lines.append(
                    f"* **{name}** — {sup} {ext_val:g} {unit} at {_fmt_ts(ext_time)} "
                    f"({n} reading{'s' if n != 1 else ''})"
                )

            prep = "above" if analysis_type == "high" else "below"
            header = (f"{len(lines)} patient{'s' if len(lines) != 1 else ''} had "
                      f"{analysis_type} {reading_type} ({prep} {threshold:g} {unit}) {period_desc}:")
            answer = header + "\n\n" + "\n".join(lines)

            if user_context is not None:
                user_context["_last_multi_patient_text"] = answer
            return answer

        except Exception as e:
            logger.error(f"Error in MultiPatientAnalysisTool: {e}")
            return json.dumps({"error": f"Error analyzing multiple patients: {str(e)}"})

    async def _arun(self, reading_type="glucose", date_filter=None, analysis_type="high",
                    custom_threshold=None, from_date=None, to_date=None):
        return self._run(reading_type, date_filter, analysis_type, custom_threshold, from_date, to_date)