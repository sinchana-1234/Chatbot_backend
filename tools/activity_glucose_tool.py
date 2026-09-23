"""
Activity -> glucose dose-response.

Buckets a patient's days by step count (low / medium / high) and compares the
average glucose on each, to answer "how does activity affect this patient's
glucose?". Activity (activity_readings.actual_time) and glucose
(glucose_readings.local_event_time) both live in Postgres on the same
patient-local time basis, so this is a clean same-DB, day-keyed correlation.

The two sides are queried separately (steps-per-day, glucose-per-day) and joined
by date in Python — no correlated subqueries, so it sidesteps SQL grouping edge
cases entirely.

The answer is built in Python and delivered verbatim (chat_routes returns it
directly, bypassing the LLM's formatting) so the wording is deterministic.

Honesty is built in: it reports the buckets as they are (it does NOT invent a
clean monotonic relationship), flags thin buckets, and says plainly when there
aren't enough overlapping days — or no overlap at all — to draw a conclusion.
"""

import logging
from typing import Optional, Any

from langchain.tools import BaseTool
from sqlalchemy import text

from dal.postgres_db import SessionLocalPG, resolve_range, date_where
from dal.database import DatabaseManager
from dal.cycle_window import cycle_window
from config import settings

logger = logging.getLogger(__name__)

# Activity bands are BUILT from the two configured step boundaries, so the label
# text can never drift from the numbers used to bucket days.
# (lo, hi, full label, short label); hi=None means open-ended. Ordered low -> high.
STEP_CUTS = [
    (0, settings.LOW_ACTIVITY_MAX_STEPS,
     f"Low activity (<{settings.LOW_ACTIVITY_MAX_STEPS:,} steps)", "low-activity"),
    (settings.LOW_ACTIVITY_MAX_STEPS, settings.HIGH_ACTIVITY_MIN_STEPS,
     f"Medium activity ({settings.LOW_ACTIVITY_MAX_STEPS:,}\u2013{settings.HIGH_ACTIVITY_MIN_STEPS:,} steps)", "medium-activity"),
    (settings.HIGH_ACTIVITY_MIN_STEPS, None,
     f"High activity (>{settings.HIGH_ACTIVITY_MIN_STEPS:,} steps)", "high-activity"),
]


def _bucket(days: list) -> list:
    """days: list of (steps, avg_glucose). Returns non-empty buckets, low->high activity."""
    out = []
    for lo, hi, label, short in STEP_CUTS:
        vals = [g for st, g in days if st is not None and st >= lo and (hi is None or st < hi)]
        if vals:
            out.append({
                "label": label,
                "short": short,
                "days": len(vals),
                "avg_glucose": round(sum(vals) / len(vals)),
            })
    return out


def _format_activity_glucose(name: Optional[str], days: list) -> str:
    """days: list of (steps, avg_glucose) for days that have BOTH."""
    name = name or "This patient"
    total = len(days)

    if total == 0:
        return (
            f"No days have both activity and glucose data for {name}, so activity's effect "
            f"on glucose can't be assessed. More activity and CGM data from the same period "
            f"is needed."
        )
    if total < settings.MIN_DAYS_TO_COMPARE:
        d = "day" if total == 1 else "days"
        return (
            f"Only {total} {d} have both activity and glucose data for {name}, which is too "
            f"little to tell how activity affects glucose. More overlapping data is needed."
        )

    buckets = _bucket(days)

    if len(buckets) < 2:
        b = buckets[0]
        return (
            f"For {name}, all {total} days with both activity and glucose data fall in the "
            f"same activity range ({b['label'].lower()}), averaging {b['avg_glucose']} mg/dL "
            f"glucose. There isn't a spread of activity levels to compare, so activity's "
            f"effect can't be assessed yet."
        )

    low_b, high_b = buckets[0], buckets[-1]
    delta = low_b["avg_glucose"] - high_b["avg_glucose"]  # +ve = higher activity, lower glucose
    based_on = f"based on {total} days with both step and glucose data"

    if abs(delta) < settings.MIN_GLUCOSE_DIFFERENCE_MGDL:
        headline = f"Activity didn't show a clear effect on {name}'s glucose, {based_on}."
    elif delta > 0:
        headline = f"{name}'s glucose levels were lower on days with higher activity, {based_on}."
    else:
        headline = f"{name}'s glucose levels were higher on days with higher activity, {based_on}."

    lines = [
        f"* **{b['label']}:** {b['days']} days \u2014 average glucose {b['avg_glucose']} mg/dL"
        for b in buckets
    ]
    out = headline + "\n\n" + "\n".join(lines)

    if abs(delta) >= settings.MIN_GLUCOSE_DIFFERENCE_MGDL:
        direction = "lower" if delta > 0 else "higher"
        overall = (
            f"Overall pattern: Average glucose was {abs(delta)} mg/dL {direction} on "
            f"{high_b['short']} days than on {low_b['short']} days."
        )
        thin = [b for b in (low_b, high_b) if b["days"] < settings.MIN_DAYS_FOR_FIRM_RESULT]
        if thin:
            if len(thin) == 1:
                overall += (
                    f"\nThe {thin[0]['short']} group includes only {thin[0]['days']} days, "
                    f"so this pattern should be interpreted with caution."
                )
            else:
                groups = " and ".join(f"{b['short']} ({b['days']} days)" for b in thin)
                overall += (
                    f"\nThe {groups} groups are small, so this pattern should be "
                    f"interpreted with caution."
                )
        out += "\n\n" + overall

    return out


def _daily_activity_glucose(patient_id: int, start=None, end=None) -> list:
    """Two day-keyed queries (steps, glucose) joined by date in Python.
    Both sides are filtered to [start, end]; when both are None it's all history."""
    act_where, act_params = date_where("actual_time", start, end, prefix="a")
    glu_where, glu_params = date_where("local_event_time", start, end, prefix="g")
    pg = SessionLocalPG()
    try:
        steps = pg.execute(text(f"""
            SELECT DATE(actual_time) AS day, SUM(total_step) AS steps
            FROM activity_readings
            WHERE patient_id = :pid AND {act_where}
            GROUP BY DATE(actual_time)
        """), {"pid": patient_id, **act_params}).mappings().all()
        glu = pg.execute(text(f"""
            SELECT DATE(local_event_time) AS day, AVG(glucose_value) AS avg_glucose
            FROM glucose_readings
            WHERE patient_id = :pid AND {glu_where}
            GROUP BY DATE(local_event_time)
        """), {"pid": patient_id, **glu_params}).mappings().all()
    finally:
        pg.close()

    glu_by_day = {g["day"]: g["avg_glucose"] for g in glu}
    days = []
    for s in steps:
        ag = glu_by_day.get(s["day"])
        if ag is not None and s["steps"] is not None:
            days.append((s["steps"], float(ag)))
    return days


class ActivityGlucoseImpactTool(BaseTool):
    """How a patient's daily activity level relates to their average glucose."""

    name: str = "get_activity_glucose_impact"
    description: str = (
        "Answer 'how does activity affect this patient's glucose?' / 'does being more active "
        "lower this patient's glucose?'. Buckets the patient's days by step count "
        "(low/medium/high) and compares average glucose across them. Requires a patient name "
        "or ID. Optional from_date/to_date ('YYYY-MM-DD', 'YYYY-MM' or 'YYYY') or a period "
        "phrase (e.g. 'last 30 days', 'July 2026') narrow the window; omit all three for the "
        "patient's full history. Returns a finished, ready-to-send summary (relay it verbatim, "
        "do not reformat)."
    )

    def __init__(self):
        super().__init__()

    def set_user_context(self, user_context):
        object.__setattr__(self, "user_context", user_context)

    def _resolve_patient(self, patient_id, patient_name):
        """Return (patient_id, display_name). Doctor-scoped for staff; own id for patients."""
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
            if not start:                      # no dates given → use current cycle
                cw = cycle_window(patient_id)
                if cw:
                    start, end = cw
                    label = f"{start} to {end}"
            days = _daily_activity_glucose(patient_id, start, end)
            answer = _format_activity_glucose(display_name, days)
            answer += f"\n\n_Period analyzed: {label}._"

            if uc is not None:
                uc["_last_activity_impact_text"] = answer
            return answer

        except Exception as e:
            logger.error(f"Error in get_activity_glucose_impact: {e}")
            return {"error": f"Error computing activity-glucose impact: {str(e)}"}