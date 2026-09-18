#!/usr/bin/env python3
"""
Specific Medical Value Tool (PostgreSQL Version - Clean)
"""

import logging
import json
from typing import Optional
from datetime import datetime
from langchain.tools import BaseTool
from sqlalchemy import text

from dal.postgres_db import SessionLocalPG

logger = logging.getLogger(__name__)


class SpecificMedicalValueTool(BaseTool):
    name: str = "get_specific_medical_value"
    description: str = (
        "Get a SPECIFIC single medical reading (glucose, BP, SpO2, heart rate, "
        "HRV, stress, or sleep) for one date or moment in time using PostgreSQL. "
        "Use for: 'what was X's glucose at 3pm', 'sleep on July 5th', 'highest "
        "reading this morning'. DO NOT use this for multi-day trends, patterns, "
        "or anything that should be shown as a chart — use get_health_progress "
        "instead for those (e.g. 'sleep quality this week', 'sleep trend', "
        "'activity over the last N days', 'time in range'). If in doubt whether "
        "the user wants one value or a trend across days, prefer "
        "get_health_progress — it covers single values too, plus a chart."
    )

    def set_user_context(self, user_context):
        object.__setattr__(self, 'user_context', user_context)

    def _run(
        self,
        patient_id: Optional[int] = None,
        patient_name: Optional[str] = None,
        reading_type: str = "glucose",
        specific_time: Optional[str] = None,
        date_filter: Optional[str] = None,
        time_range: Optional[str] = None,
        analysis_type: str = "specific"
    ) -> str:

        try:
            # 🔐 ROLE CONTROL
            user_context = getattr(self, 'user_context', None)
            if user_context and user_context.get('role_id') == 1:
                patient_id = user_context.get('user_id')
                patient_name = None
                logger.info(f"Patient access → ID {patient_id}")

            # Staff can identify the patient by name — resolve it to an ID the
            # same way UserProfileTool/DoctorPatientMappingTool do, instead of
            # silently proceeding with no patient (which previously let the
            # agent fall back to showing OTHER patients' medical data).
            elif not patient_id and patient_name:
                from dal.database import DatabaseManager
                with DatabaseManager() as db_manager:
                    # Scope to THIS doctor's own patients first — resolves most
                    # name ambiguity automatically (e.g. multiple "Vikas Reddy" in
                    # the hospital, but only one assigned to this doctor).
                    doctor_id = user_context.get('user_id') if user_context else None
                    own_patients = db_manager.get_doctor_patients(doctor_user_id=doctor_id) if doctor_id else []
                    own_matching = [
                        p for p in own_patients
                        if patient_name.lower() in f"{p.get('patient_first_name') or ''} {p.get('patient_last_name') or ''}".lower()
                    ]

                    if own_matching:
                        if len(own_matching) > 1:
                            return json.dumps({
                                "error": f"Multiple patients found matching '{patient_name}'",
                                "matching_patients": [
                                    {"id": p["patient_id"], "name": f"{p.get('patient_first_name') or ''} {p.get('patient_last_name') or ''}".strip()}
                                    for p in own_matching
                                ],
                                "suggestion": "Please specify the exact patient ID."
                            })
                        patient_id = own_matching[0]["patient_id"]
                    else:
                        # Fall back to searching all patients only if none of the
                        # doctor's own patients match (e.g. covering-for-a-colleague).
                        users = db_manager.get_users()
                        matching = [
                            u for u in users
                            if patient_name.lower() in f"{u.first_name or ''} {u.last_name or ''}".lower()
                            and u.role_id == 1
                        ]
                        if not matching:
                            return json.dumps({
                                "error": f"No patient found matching '{patient_name}'. Please check the spelling or try the full name. "
                                         f"Do not substitute another patient's data for this request."
                            })
                        if len(matching) > 1:
                            return json.dumps({
                                "error": f"Multiple patients found matching '{patient_name}'",
                                "matching_patients": [
                                    {"id": u.id, "name": f"{u.first_name or ''} {u.last_name or ''}".strip()}
                                    for u in matching
                                ],
                                "suggestion": "Please specify the exact patient ID."
                            })
                        patient_id = matching[0].id

            if not patient_id:
                return json.dumps({
                    "error": "Please specify which patient you're asking about. "
                             "Do not substitute another patient's data for this request."
                })

            db = SessionLocalPG()

            try:
                # -------------------------
                # SLEEP BRANCH
                # -------------------------
                # Sleep doesn't fit the "average a numeric column" pattern of
                # the other reading types. It lives in sleep_readings_details
                # with one row per minute of sleep, tagged by `level`:
                #   0 = deep sleep   |   1 = light sleep   |   2 = REM sleep
                #   3 = awake (not counted in total sleep)
                # We use night-attribution (actual_time + 12h) so a sleep
                # session crossing midnight is attributed to the wake-up date.
                # This mirrors the logic in dal/postgres_queries.py (REST) and
                # dal/services/medical_readings_service.py (chat sleep handler).
                if reading_type == "sleep":
                    sleep_params = {"patient_id": patient_id}

                    if date_filter:
                        if len(date_filter) == 7:
                            # YYYY-MM (month query)
                            date_condition = (
                                "TO_CHAR(actual_time + INTERVAL '12 hours', 'YYYY-MM') = :date"
                            )
                            sleep_params["date"] = date_filter
                        else:
                            # YYYY-MM-DD (single date — the "night ending on" target)
                            date_condition = (
                                "DATE(actual_time + INTERVAL '12 hours') = :date"
                            )
                            sleep_params["date"] = date_filter
                    else:
                        date_condition = "1=1"

                    sleep_row = db.execute(
                        text(f"""
                            SELECT
                                COUNT(*) FILTER (WHERE level = 0) AS deep_minutes,
                                COUNT(*) FILTER (WHERE level = 1) AS light_minutes,
                                COUNT(*) FILTER (WHERE level = 2) AS rem_minutes,
                                COUNT(*) FILTER (WHERE level = 3) AS awake_minutes,
                                COUNT(*) FILTER (WHERE level IN (0, 1, 2)) AS total_sleep_minutes
                            FROM sleep_readings_details
                            WHERE patient_id = :patient_id
                              AND {date_condition}
                        """),
                        sleep_params,
                    ).fetchone()

                    deep_min  = int(sleep_row[0] or 0)
                    light_min = int(sleep_row[1] or 0)
                    rem_min   = int(sleep_row[2] or 0)
                    awake_min = int(sleep_row[3] or 0)
                    total_min = int(sleep_row[4] or 0)

                    if total_min == 0:
                        return json.dumps({
                            "reading_type": "sleep",
                            "patient_id": patient_id,
                            "date_filter": date_filter,
                            "total_sleep_duration": "0h 0m",
                            "total_sleep_minutes": 0,
                            "sleep_breakdown": {
                                "deep_sleep_minutes": 0,
                                "light_sleep_minutes": 0,
                                "rem_sleep_minutes": 0,
                                "awake_minutes": awake_min,
                            },
                            "message": "No sleep data found for the requested period.",
                        })

                    return json.dumps({
                        "reading_type": "sleep",
                        "patient_id": patient_id,
                        "date_filter": date_filter,
                        "total_sleep_duration": f"{total_min // 60}h {total_min % 60}m",
                        "total_sleep_minutes": total_min,
                        "sleep_breakdown": {
                            "deep_sleep_minutes": deep_min,
                            "light_sleep_minutes": light_min,
                            "rem_sleep_minutes": rem_min,
                            "awake_minutes": awake_min,
                        },
                    })

                # -------------------------
                # TABLE MAPPING
                # -------------------------
                # Use each table's LOCAL time column (not UTC) for both date
                # filtering and display — using UTC caused two bugs: (1) times
                # reported to the doctor were off by the local UTC offset
                # (e.g. a 23:07 IST reading was shown as "17:37"), and (2)
                # date filters like "September 1st" could miss/misinclude
                # readings near midnight due to the UTC/local day boundary
                # mismatch. glucose_readings names its local column
                # "local_event_time"; every other reading table names it
                # "actual_time" — different names, same purpose.
                mapping = {
                    "glucose": ("glucose_readings", "glucose_value", "local_event_time"),
                    "blood_pressure": ("blood_pressure_readings", "systolic", "actual_time"),
                    "spo2": ("spo2_readings", "value", "actual_time"),
                    "body_temperature": ("body_temperature_readings", "temperature", "actual_time"),
                    "heart_rate": ("heart_rate_readings", "value", "actual_time"),
                    "hrv": ("hrv_readings", "value", "actual_time"),
                    "stress": ("stress_readings", "value", "actual_time"),
                }

                if reading_type not in mapping:
                    return f"Unsupported reading type: {reading_type}"

                table, column, time_col = mapping[reading_type]

                # -------------------------
                # DATE FILTER
                # -------------------------
                params = {"patient_id": patient_id}

                if date_filter:
                    if len(date_filter) == 7:
                        date_condition = f"TO_CHAR({time_col}, 'YYYY-MM') = :date"
                        params["date"] = date_filter
                    else:
                        date_condition = f"DATE({time_col}) = :date"
                        params["date"] = date_filter
                else:
                    date_condition = "1=1"

                # -------------------------
                # TIME RANGE FILTER
                # -------------------------
                time_condition = "1=1"
                if time_range:
                    if time_range == "morning":
                        time_condition = f"EXTRACT(HOUR FROM {time_col}) BETWEEN 6 AND 11"
                    elif time_range == "afternoon":
                        time_condition = f"EXTRACT(HOUR FROM {time_col}) BETWEEN 12 AND 16"
                    elif time_range == "evening":
                        time_condition = f"EXTRACT(HOUR FROM {time_col}) BETWEEN 17 AND 20"
                    elif time_range == "night":
                        time_condition = f"(EXTRACT(HOUR FROM {time_col}) >= 21 OR EXTRACT(HOUR FROM {time_col}) <= 5)"

                # -------------------------
                # ANALYSIS TYPE
                # -------------------------
                if analysis_type == "highest":
                    order = "DESC"
                elif analysis_type == "lowest":
                    order = "ASC"
                else:
                    order = "DESC"

                # -------------------------
                # QUERY
                # -------------------------
                query = f"""
                    SELECT {column}, {time_col}
                    FROM {table}
                    WHERE patient_id = :patient_id
                    AND {date_condition}
                    AND {time_condition}
                    ORDER BY {column} {order}
                    LIMIT 10
                """

                results = db.execute(text(query), params).fetchall()

                if not results:
                    return json.dumps({
                        "message": f"No {reading_type} readings found for this patient in the requested period.",
                        "patient_id": patient_id,
                        "note": "Report exactly this — do not substitute or display another patient's data."
                    })

                # -------------------------
                # FORMAT RESPONSE
                # -------------------------
                formatted = [
                    {
                        "value": float(r[0]) if r[0] is not None else None,
                        "time": str(r[1])
                    }
                    for r in results
                ]

                # Specific time handling
                if analysis_type == "specific" and specific_time:
                    try:
                        target_time = datetime.fromisoformat(specific_time)
                        closest = min(
                            formatted,
                            key=lambda x: abs(datetime.fromisoformat(x["time"]) - target_time)
                        )
                        return json.dumps({
                            "type": "specific",
                            "reading": closest
                        }, indent=2)
                    except Exception:
                        pass

                return json.dumps({
                    "type": analysis_type,
                    "reading_type": reading_type,
                    "count": len(formatted),
                    "results": formatted
                }, indent=2)

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error in SpecificMedicalValueTool: {e}")
            return f"Error: {str(e)}"