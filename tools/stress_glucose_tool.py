"""
Stress -> glucose correlation (per-patient median split).

Stress is a continuous 0-100 daily-average score whose values cluster tightly and
differently per patient (e.g. one patient's daily averages sit 26-43, another's
23-53). Fixed low/medium/high cutoffs would jam most days into one band and leave
others empty, so instead each patient is split at THEIR OWN median daily stress:
lower-stress days vs higher-stress days, compared against that patient's baseline.

Stress (stress_readings.value, averaged per local day) and glucose
(glucose_readings.local_event_time) both live in Postgres; queried separately by
day and joined in Python.

The direction reported is whatever the data actually shows — it does NOT assume
stress raises glucose. Answer built in Python, delivered verbatim (bypass), honest
about thin sides / no overlap / no spread, and states the date range.
"""

import logging
from statistics import median
from typing import Optional, Any

from langchain.tools import BaseTool
from sqlalchemy import text

from dal.postgres_db import SessionLocalPG, resolve_range, date_where
from dal.database import DatabaseManager
from config import settings
from dal.cycle_window import cycle_window

logger = logging.getLogger(__name__)


def _fmt_range(first, last) -> str:
    if not first or not last:
        return ""
    if first == last:
        return f" ({first:%d %b %Y})"
    return f" (between {first:%d %b %Y} and {last:%d %b %Y})"


def _format_stress_glucose(name: Optional[str], days: list, first=None, last=None) -> str:
    """days: list of (stress, avg_glucose) for days that have BOTH."""
    name = name or "This patient"
    total = len(days)
    rng = _fmt_range(first, last)

    if total == 0:
        return (
            f"There isn't enough matching data to assess how stress affects {name}'s glucose.\n\n"
            f"No days have both stress and glucose readings, so the two can't be compared. "
            f"More stress and CGM data from the same period is needed."
        )
    if total < settings.MIN_DAYS_TO_COMPARE:
        d = "day" if total == 1 else "days"
        return (
            f"There isn't enough matching data to reliably assess how stress affects {name}'s glucose.\n\n"
            f"Only {total} {d} have both stress and glucose readings, which is too few to "
            f"compare. More overlapping data is needed."
        )

    stresses = sorted(s for s, _ in days)
    med = median(stresses)
    lower = [g for s, g in days if s <= med]
    higher = [g for s, g in days if s > med]

    # If the median sits on a spike of identical values, one side can be empty.
    if not lower or not higher:
        return (
            f"There isn't enough variation in stress to assess how it affects {name}'s glucose.\n\n"
            f"Across {total} days, daily stress barely varies (around {round(med)} on a 0\u2013100 "
            f"scale), so there's no spread of stress levels to compare."
        )

    lower_g = round(sum(lower) / len(lower))
    higher_g = round(sum(higher) / len(higher))
    delta = higher_g - lower_g          # +ve = higher stress, higher glucose
    based_on = f"based on {total} days{rng} with both stress and glucose data"

    if abs(delta) < settings.MIN_GLUCOSE_DIFFERENCE_MGDL:
        headline = f"Stress didn't show a clear effect on {name}'s glucose, {based_on}."
    elif delta > 0:
        headline = f"{name}'s glucose was higher on higher-stress days, {based_on}."
    else:
        headline = f"{name}'s glucose was lower on higher-stress days, {based_on}."

    lines = [
        f"* Lower-stress days (stress at or below {round(med)}): {len(lower)} days "
        f"\u2014 average glucose {lower_g} mg/dL",
        f"* Higher-stress days (stress above {round(med)}): {len(higher)} days "
        f"\u2014 average glucose {higher_g} mg/dL",
    ]
    out = headline + "\n\n" + "\n".join(lines)

    if abs(delta) >= settings.MIN_GLUCOSE_DIFFERENCE_MGDL:
        direction = "higher" if delta > 0 else "lower"
        overall = (
            f"Overall pattern: Average glucose was {abs(delta)} mg/dL {direction} on "
            f"higher-stress days than on lower-stress days."
        )
        thin = min(len(lower), len(higher))
        if thin < settings.STRESS_THIN_SIDE_DAYS:
            side = "higher-stress" if len(higher) < len(lower) else "lower-stress"
            overall += (
                f"\n\nThe {side} group includes only {thin} days, so this pattern should be "
                f"interpreted with caution."
            )
        out += "\n\n" + overall

    return out


def _daily_stress_glucose(patient_id: int, start=None, end=None):
    """Daily avg stress joined to daily avg glucose. Returns (days, first, last).
    Both sides are filtered to [start, end]; None/None = all history."""
    stress_where, stress_params = date_where("actual_time", start, end, prefix="s")
    glu_where, glu_params = date_where("local_event_time", start, end, prefix="g")
    pg = SessionLocalPG()
    try:
        stress = pg.execute(text(f"""
            SELECT DATE(actual_time) AS day, AVG(value) AS stress
            FROM stress_readings
            WHERE patient_id = :pid AND {stress_where}
            GROUP BY DATE(actual_time)
        """), {"pid": patient_id, **stress_params}).mappings().all()
        glu = pg.execute(text(f"""
            SELECT DATE(local_event_time) AS day, AVG(glucose_value) AS avg_glucose
            FROM glucose_readings
            WHERE patient_id = :pid AND {glu_where}
            GROUP BY DATE(local_event_time)
        """), {"pid": patient_id, **glu_params}).mappings().all()
    finally:
        pg.close()

    glu_by_day = {g["day"]: g["avg_glucose"] for g in glu}
    days, present = [], []
    for s in stress:
        ag = glu_by_day.get(s["day"])
        if ag is not None and s["stress"] is not None:
            days.append((float(s["stress"]), float(ag)))
            present.append(s["day"])
    first = min(present) if present else None
    last = max(present) if present else None
    return days, first, last


class StressGlucoseImpactTool(BaseTool):
    """How a patient's daily stress relates to their glucose (per-patient median split)."""

    name: str = "get_stress_glucose_impact"
    description: str = (
        "Answer 'how does stress affect this patient's glucose?' / 'is this patient's glucose "
        "higher when stressed?'. Splits the patient's days at their own median daily stress and "
        "compares average glucose on lower- vs higher-stress days. Use ONLY for STRESS vs "
        "glucose — not activity, sleep, or meals. Requires a patient name or ID. Optional "
        "from_date/to_date ('YYYY-MM-DD', 'YYYY-MM' or 'YYYY') or a period phrase (e.g. "
        "'last 30 days', 'July 2026') narrow the window; omit all three for full history. "
        "Returns a finished, ready-to-send summary (relay it verbatim, do not reformat)."
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
            if not start:                      # no dates given → use current cycle
                cw = cycle_window(patient_id)
                if cw:
                    start, end = cw
                    label = f"{start} to {end}"
            days, first, last = _daily_stress_glucose(patient_id, start, end)
            answer = _format_stress_glucose(display_name, days, first, last)
            answer += f"\n\n_Period analyzed: {label}._"

            if uc is not None:
                uc["_last_stress_impact_text"] = answer
            return answer

        except Exception as e:
            logger.error(f"Error in get_stress_glucose_impact: {e}")
            return {"error": f"Error computing stress-glucose impact: {str(e)}"}