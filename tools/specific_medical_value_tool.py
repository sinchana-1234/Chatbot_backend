#!/usr/bin/env python3
"""
Specific Medical Value Tool (PostgreSQL Version - Clean)
"""

import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime
from langchain.tools import BaseTool
from sqlalchemy import text

from dal.postgres_db import SessionLocalPG

logger = logging.getLogger(__name__)


class SpecificMedicalValueTool(BaseTool):
    name: str = "get_specific_medical_value"
    description: str = (
        "Get a SPECIFIC single medical reading, a TRUE full-period overview, or an "
        "hour-of-day HIGH/LOW pattern for a patient's readings (glucose, BP, SpO2, "
        "heart rate, HRV, stress, or sleep) using PostgreSQL. Set analysis_type to one of: "
        "'specific' (default) — one reading at/near a given time, e.g. 'what was X's "
        "glucose at 3pm'; "
        "'overview' — for 'how is the patient's glucose', 'how is X's glucose doing', "
        "'glucose summary/status' with NO chart/trend/AGP/TIR wording — returns a REAL "
        "aggregate (count/min/max/avg computed over ALL matching rows, never a truncated "
        "slice) plus the time each extreme occurred and the date range actually covered; "
        "'pattern_high' — for 'when does glucose go high', 'when do spikes happen', "
        "'what time does X's sugar rise' — buckets every reading above a threshold "
        "(default 180 for glucose) by hour-of-day across the FULL history and reports, "
        "per hour, how many DISTINCT CALENDAR DAYS contributed a reading — an hour with "
        "distinct_days=1 is a single isolated episode, not a recurring pattern, and must "
        "be reported as such; "
        "'pattern_low' — same as pattern_high but for readings below a threshold "
        "(default 70 for glucose), for 'when does glucose go low', 'when do drops/lows "
        "happen'. "
        "An optional threshold parameter overrides the default cutoff for pattern_high/"
        "pattern_low. "
        "DO NOT use this for multi-day trend CHARTS — use get_health_progress instead "
        "for those (e.g. 'sleep quality this week', 'sleep trend', 'activity over the "
        "last N days', 'time in range chart'). If the user explicitly wants a chart, "
        "prefer get_health_progress; if they want a plain-language status/pattern answer "
        "with no chart, use this tool's overview/pattern_high/pattern_low modes."
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
        analysis_type: str = "specific",
        threshold: Optional[float] = None
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
                # OVERVIEW — true full-period aggregate, NEVER a truncated slice.
                # Fixes the bug where "how is glucose" relabeled a value from a
                # 10-row ORDER BY...LIMIT 10 slice as the real min/max.
                # -------------------------
                if analysis_type == "overview":
                    agg_row = db.execute(
                        text(f"""
                            SELECT
                                COUNT(*) AS cnt,
                                MIN({column}) AS min_val,
                                MAX({column}) AS max_val,
                                AVG({column}) AS avg_val,
                                MIN({time_col}) AS earliest,
                                MAX({time_col}) AS latest
                            FROM {table}
                            WHERE patient_id = :patient_id
                            AND {date_condition}
                            AND {time_condition}
                        """),
                        params,
                    ).fetchone()

                    cnt = int(agg_row[0] or 0)
                    if cnt == 0:
                        return json.dumps({
                            "message": f"No {reading_type} readings found for this patient in the requested period.",
                            "patient_id": patient_id,
                            "note": "Report exactly this — do not substitute or display another patient's data."
                        })

                    # Fetch the actual reading (with its timestamp) at the true
                    # min and max — not from a limited/pre-sorted slice, but a
                    # fresh, targeted lookup so the reported extremes are real.
                    min_reading = db.execute(
                        text(f"""
                            SELECT {column}, {time_col} FROM {table}
                            WHERE patient_id = :patient_id AND {date_condition} AND {time_condition}
                            ORDER BY {column} ASC, {time_col} ASC LIMIT 1
                        """),
                        params,
                    ).fetchone()
                    max_reading = db.execute(
                        text(f"""
                            SELECT {column}, {time_col} FROM {table}
                            WHERE patient_id = :patient_id AND {date_condition} AND {time_condition}
                            ORDER BY {column} DESC, {time_col} ASC LIMIT 1
                        """),
                        params,
                    ).fetchone()

                    return json.dumps({
                        "type": "overview",
                        "reading_type": reading_type,
                        "patient_id": patient_id,
                        "total_readings_in_period": cnt,
                        "average": round(float(agg_row[3]), 1) if agg_row[3] is not None else None,
                        "lowest": {"value": float(min_reading[0]), "time": str(min_reading[1])},
                        "highest": {"value": float(max_reading[0]), "time": str(max_reading[1])},
                        "period_covered": {"from": str(agg_row[4]), "to": str(agg_row[5])},
                        "note": (
                            "This is a real aggregate computed over ALL matching readings "
                            "in the period (no row limit) — count, average, lowest, and "
                            "highest are all authoritative, not derived from a partial slice."
                        )
                    }, indent=2)

                # -------------------------
                # PATTERN_HIGH / PATTERN_LOW — hour-of-day bucketing across the
                # FULL matching history, with distinct-day counts per hour so a
                # single night's run of extreme readings can never be reported
                # as a recurring daily pattern.
                # -------------------------
                if analysis_type in ("pattern_high", "pattern_low"):
                    default_thresholds = {
                        "glucose": {"pattern_high": 180, "pattern_low": 70},
                        "blood_pressure": {"pattern_high": 140, "pattern_low": 90},
                        "spo2": {"pattern_high": 100, "pattern_low": 92},
                        "heart_rate": {"pattern_high": 100, "pattern_low": 60},
                    }
                    effective_threshold = threshold
                    if effective_threshold is None:
                        effective_threshold = default_thresholds.get(reading_type, {}).get(analysis_type)
                    if effective_threshold is None:
                        return json.dumps({
                            "error": f"No default {analysis_type} threshold is defined for reading_type "
                                     f"'{reading_type}'. Please pass an explicit threshold."
                        })

                    comparison = ">=" if analysis_type == "pattern_high" else "<="

                    rows = db.execute(
                        text(f"""
                            SELECT
                                {column} AS val,
                                {time_col} AS ts,
                                EXTRACT(HOUR FROM {time_col})::int AS hr,
                                DATE({time_col}) AS day
                            FROM {table}
                            WHERE patient_id = :patient_id
                            AND {date_condition}
                            AND {time_condition}
                            AND {column} {comparison} :threshold
                        """),
                        {**params, "threshold": effective_threshold},
                    ).fetchall()

                    if not rows:
                        return json.dumps({
                            "message": (
                                f"No {reading_type} readings {comparison} {effective_threshold} "
                                f"found for this patient in the requested period."
                            ),
                            "patient_id": patient_id,
                            "threshold": effective_threshold
                        })

                    buckets: Dict[int, Dict[str, Any]] = {}
                    all_days = set()
                    for val, ts, hr, day in rows:
                        hr = int(hr)
                        all_days.add(day)
                        b = buckets.setdefault(hr, {"count": 0, "days": set(), "examples": []})
                        b["count"] += 1
                        b["days"].add(day)
                        if len(b["examples"]) < 3:
                            b["examples"].append({"value": float(val), "time": str(ts)})

                    hourly_pattern = [
                        {
                            "hour_of_day": hr,
                            "count": b["count"],
                            "distinct_days": len(b["days"]),
                            "example_readings": b["examples"],
                        }
                        for hr, b in sorted(buckets.items(), key=lambda kv: -kv[1]["count"])
                    ]

                    return json.dumps({
                        "type": analysis_type,
                        "reading_type": reading_type,
                        "threshold": effective_threshold,
                        "total_matching_readings": len(rows),
                        "total_distinct_days_with_data": len(all_days),
                        "hourly_pattern": hourly_pattern,
                        "interpretation_note": (
                            "Only describe a genuine time-of-day PATTERN for hours where "
                            "distinct_days is 2 or more (that hour showed extreme readings on "
                            "multiple different days). An hour with distinct_days=1 means every "
                            "matching reading in that bucket came from a single day/night — "
                            "report that as an ISOLATED EPISODE on that specific date, never as "
                            "a recurring daily pattern."
                        )
                    }, indent=2)

                # -------------------------
                # ANALYSIS TYPE (specific / highest / lowest — unchanged legacy behavior)
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