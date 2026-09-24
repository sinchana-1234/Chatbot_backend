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
from dal.cycle_window import cycle_window

logger = logging.getLogger(__name__)

# --- Deterministic formatter for pattern_high / pattern_low answers ------------
# Built in Python and delivered verbatim (same bypass the glucose-trend tool uses)
# so the model can't re-bullet or reword it. The band map below is the one tunable
# knob for the descriptive sentences.
_PATTERN_BANDS = [
    (range(0, 5),   "overnight"),
    (range(5, 9),   "early morning"),
    (range(9, 12),  "late morning"),
    (range(12, 17), "afternoon"),
    (range(17, 20), "evening"),
    (range(20, 24), "late evening"),
]
_READING_LABEL = {
    "glucose": "glucose",
    "blood_pressure": "blood pressure",
    "spo2": "SpO2",
    "heart_rate": "heart rate",
}

def _pattern_band_of(h):
    for rng, lbl in _PATTERN_BANDS:
        if h in rng:
            return lbl
    return "other"

def _pattern_h12(h):
    return f"{12 if h % 12 == 0 else h % 12} {'AM' if h < 12 else 'PM'}"

def _pattern_hour_range(h):
    return f"{_pattern_h12(h)}–{_pattern_h12((h + 1) % 24)}"

def _pattern_join(items):
    items = list(dict.fromkeys(items))
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"

def _pattern_top_bands(bands_seen, n):
    counts, order = {}, []
    for b in bands_seen:
        if b not in counts:
            counts[b] = 0
            order.append(b)
        counts[b] += 1
    order.sort(key=lambda b: -counts[b])  # stable: ties keep first-seen order
    return order[:n]

def _format_pattern_prose(patient_name, reading_type, analysis_type, hourly_pattern):
    is_high = analysis_type == "pattern_high"
    adj = "elevated" if is_high else "low"
    Adj = "Elevated" if is_high else "Low"
    label = _READING_LABEL.get(reading_type, reading_type.replace("_", " "))
    name = patient_name or "This patient"

    # Clinical-safety rule enforced in CODE (was a prompt instruction): only call
    # an hour a recurring PATTERN if it recurs on 2+ distinct days.
    recurring = sorted(
        (e for e in hourly_pattern if e.get("distinct_days", 0) >= 2),
        key=lambda e: e.get("distinct_days", 0),
        reverse=True,
    )
    if not recurring:
        prep = "above" if is_high else "below"
        return (f"{name}'s {label} readings {prep} the threshold look like "
                f"isolated single-day episodes, not a recurring time-of-day pattern.")
    top = recurring[:5]

    lead_bands = _pattern_top_bands([_pattern_band_of(e["hour_of_day"]) for e in top], 2)
    lead = (f"{name}'s {label} levels are most frequently {adj} during the "
            f"{_pattern_join(lead_bands)} hours.")

    bullets = [
        f"* **{_pattern_hour_range(e['hour_of_day'])}** — {Adj} on {e['distinct_days']} days"
        for e in top
    ]

    h0 = top[0]["hour_of_day"]
    peak = "around midnight" if h0 in (23, 0, 1) else f"in the {_pattern_band_of(h0)} hours"
    rest = [_pattern_band_of(e["hour_of_day"]) for e in top[1:]
            if _pattern_band_of(e["hour_of_day"]) != _pattern_band_of(h0)]
    overall = (f"Overall pattern: {Adj} {label} occurs most frequently {peak}"
               + (f" and during the {_pattern_join(rest)} hours." if rest else "."))

    return lead + "\nThe main periods are:\n\n" + "\n".join(bullets) + "\n\n" + overall

def _friendly_dt(ts):
    """Render a reading timestamp for humans:
    '2026-07-13 00:00:00' -> 'July 13, 2026 at 12:00 AM'.
    Midnight becomes '12:00 AM', not the raw '00:00:00' the model was
    echoing. Windows-safe (avoids %-d / %-I, which crash on Windows)."""
    if ts is None:
        return None
    if not isinstance(ts, datetime):
        try:
            ts = datetime.fromisoformat(str(ts))
        except Exception:
            return str(ts)
    hour = ts.hour
    suffix = "AM" if hour < 12 else "PM"
    hour12 = 12 if hour % 12 == 0 else hour % 12
    clock = f"{hour12}:{ts.minute:02d} {suffix}"
    return ts.strftime("%B ") + str(ts.day) + f", {ts.year} at {clock}"


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
        "DO NOT use this for multi-day trend CHARTS — use the get_*_trend tools instead "
        "(get_glucose_trend, get_tir_trend, get_sleep_trend, get_activity_trend, etc.) "
        "for those (e.g. 'sleep quality this week', 'sleep trend', 'activity over the "
        "last N days', 'time in range chart'). If the user explicitly wants a chart, "
        "prefer the matching get_*_trend tool; if they want a plain-language status/pattern answer "
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
                        cw = cycle_window(patient_id)
                        if cw:
                            date_condition = "DATE(actual_time + INTERVAL '12 hours') BETWEEN :cstart AND :cend"
                            sleep_params["cstart"], sleep_params["cend"] = cw
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

                # Blood pressure stores two paired values on one row. The primary
                # column (systolic) drives ordering; the secondary (diastolic) is
                # fetched alongside and shown as "systolic/diastolic", so a BP
                # reading is never reported as a single number.
                secondary_col = {"blood_pressure": "diastolic"}.get(reading_type)

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
                    # No date given → default to the patient's current program
                    # cycle. An explicit date_filter above always overrides this.
                    cw = cycle_window(patient_id)
                    if cw:
                        date_condition = f"DATE({time_col}) BETWEEN :cstart AND :cend"
                        params["cstart"], params["cend"] = cw
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

                    # Format period dates
                    period_from = str(agg_row[4])
                    period_to = str(agg_row[5])

                    # Reading type labels for display
                    reading_labels = {
                        "glucose": "glucose",
                        "blood_pressure": "blood pressure",
                        "heart_rate": "heart rate",
                        "spo2": "SpO2",
                        "stress": "stress",
                        "hrv": "HRV",
                    }

                    reading_units = {
                        "glucose": "mg/dL",
                        "blood_pressure": "mmHg",
                        "heart_rate": "bpm",
                        "spo2": "%",
                        "stress": "%",
                        "hrv": "ms",
                    }

                    label = reading_labels.get(reading_type, reading_type.replace("_", " "))
                    unit = reading_units.get(reading_type, "")

                    # Format the prose response deterministically
                    overview_prose = (
                        f"* **Period Covered:** {period_from} to {period_to}\n"
                        f"* **Average {label.title()}:** {round(float(agg_row[3]), 1) if agg_row[3] is not None else 'N/A'} {unit}\n"
                        f"* **Lowest {label.title()}:** {float(min_reading[0])} {unit} (recorded on {min_reading[1]})\n"
                        f"* **Highest {label.title()}:** {float(max_reading[0])} {unit} (recorded on {max_reading[1]})\n"
                        f"* **Total Readings in Period:** {cnt}"
                    )

                    if user_context is not None:
                        user_context['_last_overview_text'] = overview_prose
                    return overview_prose

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

                    # Build the answer deterministically and deliver it verbatim
                    # (bypass), so the model can't re-bullet or reword it.
                    patient_display_name = None
                    try:
                        from dal.database import DatabaseManager
                        with DatabaseManager() as _dm:
                            _u = _dm.get_users(user_id=patient_id)
                            if _u:
                                patient_display_name = f"{_u[0].first_name or ''} {_u[0].last_name or ''}".strip() or None
                    except Exception:
                        patient_display_name = None

                    pattern_text = _format_pattern_prose(
                        patient_display_name, reading_type, analysis_type, hourly_pattern
                    )
                    if user_context is not None:
                        user_context['_last_pattern_text'] = pattern_text
                    return pattern_text

                # -------------------------
                # SPECIFIC TIME — nearest reading to the requested time across ALL
                # readings, NOT a value-ranked slice. (The old path did
                # ORDER BY {value} DESC LIMIT 10 then picked the nearest-in-time among
                # those 10, so the true reading at the asked time was missed whenever it
                # wasn't one of the day's 10 highest — e.g. "10 AM" returned a 250 spike
                # instead of the real 80. Fixed: order by absolute time distance, LIMIT 1.)
                # -------------------------
                if analysis_type == "specific" and specific_time:
                    try:
                        target_dt = datetime.fromisoformat(specific_time)
                    except Exception:
                        target_dt = None
                    if target_dt is not None:
                        near_select = f"{column}, {secondary_col}, {time_col}" if secondary_col else f"{column}, {time_col}"
                        near = db.execute(
                            text(f"""
                                SELECT {near_select}
                                FROM {table}
                                WHERE patient_id = :patient_id
                                AND {date_condition}
                                AND {time_condition}
                                ORDER BY ABS(EXTRACT(EPOCH FROM ({time_col} - :target)))
                                LIMIT 1
                            """),
                            {**params, "target": target_dt},
                        ).fetchone()
                        if not near:
                            return json.dumps({
                                "message": f"No {reading_type} readings found for this patient in the requested period.",
                                "patient_id": patient_id,
                                "note": "Report exactly this — do not substitute or display another patient's data."
                            })
                        if secondary_col:
                            reading = {
                                "value": f"{int(near[0])}/{int(near[1])}" if near[0] is not None and near[1] is not None else None,
                                "unit": "mmHg",
                                "time": str(near[2]),
                            }
                        else:
                            reading = {
                                "value": float(near[0]) if near[0] is not None else None,
                                "time": str(near[1]),
                            }
                        return json.dumps({
                            "type": "specific",
                            "reading": reading,
                        }, indent=2)

                # -------------------------
                # PLAIN VALUE QUESTION, NO TIME → DAY SUMMARY (not the peak)
                # "what is my stress level on 21 July" has many readings that day;
                # the old DESC fall-through returned only the highest, implying the
                # peak was "the" level. Return an honest count/min/max/avg summary.
                # -------------------------
                if analysis_type == "specific" and not specific_time:
                    agg = db.execute(
                        text(f"""
                            SELECT COUNT(*), MIN({column}), MAX({column}), AVG({column})
                            FROM {table}
                            WHERE patient_id = :patient_id AND {date_condition} AND {time_condition}
                        """),
                        params,
                    ).fetchone()
                    cnt = int(agg[0] or 0)
                    if cnt == 0:
                        return json.dumps({
                            "message": f"No {reading_type} readings found for this patient in the requested period.",
                            "patient_id": patient_id,
                            "note": "Report exactly this — do not substitute or display another patient's data."
                        })
                    return json.dumps({
                        "type": "day_summary",
                        "reading_type": reading_type,
                        "count": cnt,
                        "average": round(float(agg[3]), 1) if agg[3] is not None else None,
                        "lowest": float(agg[1]),
                        "highest": float(agg[2]),
                        "note": (
                            "This period has multiple readings, so there is no single value. "
                            "Report it as a summary — the average plus the low-to-high range — "
                            "NOT just the highest. Never present the peak as the patient's level."
                        )
                    }, indent=2)

                # -------------------------
                # ANALYSIS TYPE (highest / lowest / specific-without-time)
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
                # Tie-break by time so a tied extreme (e.g. several 80s on one
                # day) always resolves to the SAME reading — the earliest one —
                # instead of an arbitrary row. Without this, "lowest = 80" could
                # report a different timestamp each run and never match the
                # verify SQL (which uses ORDER BY value, time ASC). For "highest"
                # (DESC) the earliest occurrence still reads most naturally, so
                # time stays ASC in both cases.
                select_cols = f"{column}, {secondary_col}, {time_col}" if secondary_col else f"{column}, {time_col}"
                query = f"""
                    SELECT {select_cols}
                    FROM {table}
                    WHERE patient_id = :patient_id
                    AND {date_condition}
                    AND {time_condition}
                    ORDER BY {column} {order}, {time_col} ASC
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
                if secondary_col:
                    formatted = [
                        {
                            "value": f"{int(r[0])}/{int(r[1])}" if r[0] is not None and r[1] is not None else None,
                            "unit": "mmHg",
                            "time": _friendly_dt(r[2]),
                        }
                        for r in results
                    ]
                else:
                    formatted = [
                        {
                            "value": float(r[0]) if r[0] is not None else None,
                            "time": _friendly_dt(r[1])   # 'July 13, 2026 at 12:00 AM', not '00:00:00'
                        }
                        for r in results
                    ]

                # For highest/lowest the answer is ONE reading — the top row after
                # the ORDER BY. Surface it as a single authoritative field so the
                # model has no list to mis-read and no reason to substitute a
                # clinical threshold number (the "lowest = 70" hallucination:
                # 70 is the Low<70 cutoff, not a real reading — real min was 51).
                answer = formatted[0] if formatted else None

                if secondary_col:
                    note = (
                        f"The {analysis_type} blood pressure reading is EXACTLY "
                        f"answer.value (systolic/diastolic, mmHg) at answer.time. "
                        f"Report that systolic/diastolic pair verbatim."
                    )
                else:
                    note = (
                        f"The {analysis_type} {reading_type} reading is EXACTLY "
                        f"answer.value at answer.time. Report that number verbatim. "
                        f"Do NOT report any normal-range boundary or clinical "
                        f"threshold (e.g. 70/140/180) as the value."
                    )
                return json.dumps({
                    "type": analysis_type,
                    "reading_type": reading_type,
                    "answer": answer,
                    "note": note,
                    "count": len(formatted),
                    "results": formatted
                }, indent=2)

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error in SpecificMedicalValueTool: {e}")
            return f"Error: {str(e)}"