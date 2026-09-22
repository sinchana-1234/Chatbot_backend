"""
Sleep -> glucose dose-response.

Buckets a patient's nights by total sleep duration (short / adequate / long) and
compares the *next day's* average glucose on each, to answer "how does sleep
affect this patient's glucose?".

Sleep hours come from Postgres `sleep_readings_details` — one row per minute-level
sample, level IN (0,1,2) meaning asleep (light/deep/REM), counted and divided by
60. A night is attributed to its WAKE-UP date via `actual_time + INTERVAL '12
hours'` (the codebase's standard, from postgres_queries.py), and matched to that
same date's glucose. Both sides live in Postgres on a patient-local time basis, so
this is a clean same-DB, day-keyed correlation queried separately and joined in
Python.

Answer built in Python, delivered verbatim (chat_routes bypass), honest about thin
buckets / no overlap / no spread, and states the date range it covers.
"""

import logging
from typing import Optional, Any

from langchain.tools import BaseTool
from sqlalchemy import text

from dal.postgres_db import SessionLocalPG, resolve_range, date_where
from config import settings
from dal.database import DatabaseManager

logger = logging.getLogger(__name__)

# --- tunable knobs ---
# (lo_hours, hi_hours, full label, short label); hi=None = open-ended. Low -> high.
# Sleep bands are BUILT from the two configured hour boundaries, so the label
# text can never drift from the numbers used to bucket nights.
# (lo_hours, hi_hours, full label, short label); hi=None = open-ended. Low -> high.
SLEEP_CUTS = [
    (0, settings.SLEEP_SHORT_MAX_HOURS,
     f"Short sleep (<{settings.SLEEP_SHORT_MAX_HOURS} h)", "short-sleep"),
    (settings.SLEEP_SHORT_MAX_HOURS, settings.SLEEP_LONG_MIN_HOURS,
     f"Adequate sleep ({settings.SLEEP_SHORT_MAX_HOURS}\u2013{settings.SLEEP_LONG_MIN_HOURS} h)", "adequate-sleep"),
    (settings.SLEEP_LONG_MIN_HOURS, None,
     f"Long sleep (>{settings.SLEEP_LONG_MIN_HOURS} h)", "long-sleep"),
]


def _fmt_range(first, last) -> str:
    if not first or not last:
        return ""
    if first == last:
        return f" ({first:%d %b %Y})"
    return f" (between {first:%d %b %Y} and {last:%d %b %Y})"


def _bucket(nights: list) -> list:
    """nights: list of (hours, avg_glucose). Returns non-empty buckets, short->long."""
    out = []
    for lo, hi, label, short in SLEEP_CUTS:
        vals = [g for h, g in nights if h is not None and h >= lo and (hi is None or h < hi)]
        if vals:
            out.append({"label": label, "short": short, "days": len(vals),
                        "avg_glucose": round(sum(vals) / len(vals))})
    return out


def _format_sleep_glucose(name: Optional[str], nights: list, first=None, last=None) -> str:
    """nights: list of (sleep_hours, next_day_avg_glucose) for nights that have BOTH."""
    name = name or "This patient"
    total = len(nights)
    rng = _fmt_range(first, last)

    if total == 0:
        return (
            f"There isn't enough matching data to assess how sleep affects {name}'s glucose.\n\n"
            f"No nights have both sleep and next-day glucose readings, so the two can't be "
            f"compared. More sleep and CGM data from the same period is needed."
        )
    if total <  settings.MIN_DAYS_TO_COMPARE:
        n = "night" if total == 1 else "nights"
        return (
            f"There isn't enough matching data to reliably assess how sleep affects {name}'s glucose.\n\n"
            f"Only {total} {n} have both sleep and next-day glucose readings, which is too few "
            f"to compare. More overlapping data is needed."
        )

    buckets = _bucket(nights)
    if len(buckets) < 2:
        b = buckets[0]
        return (
            f"There isn't enough variation in sleep to assess how it affects {name}'s glucose.\n\n"
            f"All {total} nights with both sleep and glucose data fall in the same range "
            f"({b['label'].lower()}), with next-day glucose averaging {b['avg_glucose']} mg/dL, "
            f"so there's no spread of sleep durations to compare."
        )

    short_b, long_b = buckets[0], buckets[-1]
    # +ve delta = longer sleep, lower next-day glucose
    delta = short_b["avg_glucose"] - long_b["avg_glucose"]
    based_on = f"based on {total} nights{rng} with both sleep and next-day glucose data"

    if abs(delta) < settings.MIN_GLUCOSE_DIFFERENCE_MGDL:
        headline = f"Sleep duration didn't show a clear effect on {name}'s glucose, {based_on}."
    elif delta > 0:
        headline = f"{name}'s next-day glucose was lower after longer sleep, {based_on}."
    else:
        headline = f"{name}'s next-day glucose was higher after longer sleep, {based_on}."

    lines = [
        f"* {b['label']}: {b['days']} nights \u2014 next-day glucose {b['avg_glucose']} mg/dL"
        for b in buckets
    ]
    out = headline + "\n\n" + "\n".join(lines)

    if abs(delta) >= settings.MIN_GLUCOSE_DIFFERENCE_MGDL:
        direction = "lower" if delta > 0 else "higher"
        overall = (
            f"Overall pattern: Next-day glucose averaged {abs(delta)} mg/dL {direction} after "
            f"{long_b['short'].replace('-', ' ')} nights than after {short_b['short'].replace('-', ' ')} nights."
        )
        thin = [b for b in (short_b, long_b) if b["days"] < settings.MIN_DAYS_FOR_FIRM_RESULT]
        if thin:
            if len(thin) == 1:
                overall += (
                    f"\n\nThe {thin[0]['short'].replace('-', ' ')} group includes only "
                    f"{thin[0]['days']} nights, so this pattern should be interpreted with caution."
                )
            else:
                groups = " and ".join(f"{b['short'].replace('-', ' ')} ({b['days']} nights)" for b in thin)
                overall += (
                    f"\n\nThe {groups} groups are small, so this pattern should be "
                    f"interpreted with caution."
                )
        out += "\n\n" + overall

    return out


def _nightly_sleep_glucose(patient_id: int, start=None, end=None):
    """Nightly sleep hours (wake-date) joined to that date's glucose. Returns (nights, first, last).
    Both sides are filtered to [start, end] on the wake-up date; None/None = all history."""
    sleep_where, sleep_params = date_where("actual_time + INTERVAL '12 hours'", start, end, prefix="s")
    glu_where, glu_params = date_where("local_event_time", start, end, prefix="g")
    pg = SessionLocalPG()
    try:
        sleep = pg.execute(text(f"""
            SELECT DATE(actual_time + INTERVAL '12 hours') AS day,
                   COUNT(*) / 60.0 AS hours
            FROM sleep_readings_details
            WHERE patient_id = :pid AND level IN (0, 1, 2) AND {sleep_where}
            GROUP BY DATE(actual_time + INTERVAL '12 hours')
        """), {"pid": patient_id, **sleep_params}).mappings().all()
        glu = pg.execute(text(f"""
            SELECT DATE(local_event_time) AS day, AVG(glucose_value) AS avg_glucose
            FROM glucose_readings
            WHERE patient_id = :pid AND {glu_where}
            GROUP BY DATE(local_event_time)
        """), {"pid": patient_id, **glu_params}).mappings().all()
    finally:
        pg.close()

    glu_by_day = {g["day"]: g["avg_glucose"] for g in glu}
    nights, days_present = [], []
    for s in sleep:
        ag = glu_by_day.get(s["day"])
        if ag is not None and s["hours"] is not None:
            nights.append((float(s["hours"]), float(ag)))
            days_present.append(s["day"])
    first = min(days_present) if days_present else None
    last = max(days_present) if days_present else None
    return nights, first, last


class SleepGlucoseImpactTool(BaseTool):
    """How a patient's nightly sleep duration relates to their next-day glucose."""

    name: str = "get_sleep_glucose_impact"
    description: str = (
        "Answer 'how does sleep affect this patient's glucose?' / 'does sleeping more/less "
        "change this patient's glucose?'. Buckets nights by sleep duration (short/adequate/long) "
        "and compares next-day average glucose. Use ONLY for SLEEP vs glucose — not activity, "
        "stress, or meals. Requires a patient name or ID. Optional from_date/to_date "
        "('YYYY-MM-DD', 'YYYY-MM' or 'YYYY') or a period phrase (e.g. 'last 30 days', "
        "'July 2026') narrow the window; omit all three for full history. Returns a finished, "
        "ready-to-send summary (relay it verbatim, do not reformat)."
    )

    def __init__(self):
        super().__init__()

    def set_user_context(self, user_context):
        object.__setattr__(self, "user_context", user_context)

    def _resolve_patient(self, patient_id, patient_name):
        uc = getattr(self, "user_context", None)
        if uc and uc.get("role_id") == 1:
            patient_id = uc.get("user_id")
        with DatabaseManager() as dm:
            if patient_id is None and patient_name:
                doctor_id = uc.get("user_id") if uc else None
                patients = dm.get_doctor_patients(doctor_user_id=doctor_id, active_only=True) if doctor_id else []
                name_l = patient_name.lower().strip()
                for p in patients:
                    full = f"{p.get('patient_first_name', '')} {p.get('patient_last_name', '')}".lower().strip()
                    if name_l in full:
                        return p["patient_id"], full.title() or None
                return None, None
            display = None
            if patient_id is not None:
                users = dm.get_users(user_id=patient_id)
                if users:
                    display = f"{users[0].first_name or ''} {users[0].last_name or ''}".strip() or None
            return patient_id, display

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
             from_date: Optional[str] = None, to_date: Optional[str] = None,
             period: Optional[str] = None) -> Any:
        try:
            uc = getattr(self, "user_context", None)
            if (patient_id is None and not patient_name
                    and not (uc and uc.get("role_id") == 1)):
                return {"error": "Please specify a patient name or ID."}

            patient_id, display_name = self._resolve_patient(patient_id, patient_name)
            if not patient_id:
                return {"error": "Could not resolve that patient among your patients."}

            start, end, label = resolve_range(from_date, to_date, period)
            nights, first, last = _nightly_sleep_glucose(patient_id, start, end)
            answer = _format_sleep_glucose(display_name, nights, first, last)
            answer += f"\n\n_Period analyzed: {label}._"

            if uc is not None:
                uc["_last_sleep_impact_text"] = answer
            return answer

        except Exception as e:
            logger.error(f"Error in get_sleep_glucose_impact: {e}")
            return {"error": f"Error computing sleep-glucose impact: {str(e)}"}