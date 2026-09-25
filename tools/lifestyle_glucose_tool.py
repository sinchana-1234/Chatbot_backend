"""
Combined lifestyle -> glucose tool.

Answers questions that name MORE THAN ONE lifestyle factor at once, e.g.
"how do sleep and stress relate to this patient's glucose?". The individual
tools (get_sleep_glucose_impact, get_stress_glucose_impact,
get_activity_glucose_impact) each answer a single factor and each own one
verbatim-bypass slot, so a compound question can only ever deliver one of them.
This tool computes each REQUESTED factor and emits ONE combined block, with an
explicit "not available" line for any requested factor that has no data.

It reuses the already-validated correlation functions from the single-factor
tools — no new correlation math lives here, only section assembly.
"""

import logging
from typing import Optional, Any, List

from langchain.tools import BaseTool

from dal.database import DatabaseManager
from dal.cycle_window import cycle_window
from dal.postgres_db import resolve_range

# Reuse the exact validated correlation + formatting from the single-factor tools.
from tools.sleep_glucose_tool import _nightly_sleep_glucose, _format_sleep_glucose
from tools.stress_glucose_tool import _daily_stress_glucose, _format_stress_glucose
from tools.activity_glucose_tool import _daily_activity_glucose, _format_activity_glucose

from datetime import datetime

def _format_iso_date(date_str):
    """Convert YYYY-MM-DD to DD-MM-YYYY"""
    try:
        date_obj = datetime.strptime(str(date_str), "%Y-%m-%d")
        return date_obj.strftime("%d-%m-%Y")
    except Exception:
        return str(date_str)

logger = logging.getLogger(__name__)

# canonical factor name -> (section heading, "not available" label)
_FACTOR_META = {
    "sleep":    ("Sleep",    "Sleep data was not available for this analysis."),
    "stress":   ("Stress",   "Stress data was not available for this analysis."),
    "activity": ("Activity", "Activity data was not available for this analysis."),
}
_VALID_FACTORS = list(_FACTOR_META.keys())


def _factor_phrase(factors: List[str]) -> str:
    """'sleep and stress' / 'sleep, stress, and activity' / 'stress'."""
    words = {"sleep": "sleep duration", "stress": "stress", "activity": "activity"}
    parts = [words[f] for f in factors]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _section(name: str, factor: str, patient_id: int, start=None, end=None) -> str:
    """Run one factor's correlation and return its formatted block (already prose).
    The [start, end] range is passed through to each factor's query; None/None = all history."""
    if factor == "sleep":
        nights, first, last = _nightly_sleep_glucose(patient_id, start, end)
        return _format_sleep_glucose(name, nights, first, last)
    if factor == "stress":
        days, first, last = _daily_stress_glucose(patient_id, start, end)
        return _format_stress_glucose(name, days, first, last)
    # activity
    days = _daily_activity_glucose(patient_id, start, end)
    return _format_activity_glucose(name, days)


def _format_combined(name: Optional[str], factors: List[str], patient_id: int,
                     start=None, end=None) -> str:
    name = name or "This patient"
    header = (
        f"{name}'s available data shows how {_factor_phrase(factors)} relate to their glucose."
    )
    blocks = [header]
    for f in factors:
        heading, _ = _FACTOR_META[f]
        body = _section(name, f, patient_id, start, end)
        blocks.append(f"### {heading}\n\n{body}")
    return "\n\n".join(blocks)


class LifestyleGlucoseImpactTool(BaseTool):
    """Combined sleep / stress / activity vs glucose, for questions naming 2+ factors."""

    name: str = "get_lifestyle_glucose_impact"
    description: str = (
        "Use this ONLY when a question asks about TWO OR MORE lifestyle factors together vs "
        "glucose — e.g. 'how do sleep and stress relate to this patient's glucose?', 'how do "
        "sleep, stress and activity affect glucose?', or a general 'how do lifestyle factors "
        "affect this patient's glucose?'. Pass the factors named in the question via the "
        "'factors' argument (any of: sleep, stress, activity); for a general 'lifestyle "
        "factors' question with none named, pass all three. For a SINGLE factor, use the "
        "dedicated tool instead (get_sleep_glucose_impact / get_stress_glucose_impact / "
        "get_activity_glucose_impact). Requires a patient name or ID. Optional from_date/to_date "
        "('YYYY-MM-DD', 'YYYY-MM' or 'YYYY') or a period phrase (e.g. 'last 30 days', 'July 2026') "
        "narrow the window; omit all three for full history. Returns a finished, ready-to-send "
        "summary (relay it verbatim, do not reformat)."
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
             factors: Optional[List[str]] = None,
             from_date: Optional[str] = None, to_date: Optional[str] = None,
             period: Optional[str] = None) -> Any:
        try:
            uc = getattr(self, "user_context", None)
            if (patient_id is None and not patient_name
                    and not (uc and uc.get("role_id") == 1)):
                return {"error": "Please specify a patient name or ID."}

            # Normalise requested factors; default to all three if unspecified.
            requested = [f.lower().strip() for f in (factors or [])]
            requested = [f for f in requested if f in _VALID_FACTORS]
            # de-dupe, keep a stable order: sleep, stress, activity
            requested = [f for f in _VALID_FACTORS if f in requested]
            if not requested:
                requested = list(_VALID_FACTORS)

            patient_id, display_name = self._resolve_patient(patient_id, patient_name)
            if not patient_id:
                return {"error": "Could not resolve that patient among your patients."}

            start, end, label = resolve_range(from_date, to_date, period)
            if not start:                      # no dates given → use current cycle
                cw = cycle_window(patient_id)
                if cw:
                    start, end = cw
                    label = f"{_format_iso_date(start)} to {_format_iso_date(end)}"  # Returns: "11-09-2026 to 25-09-2026"
            answer = _format_combined(display_name, requested, patient_id, start, end)
            answer += f"\n\n_Period analyzed: {label}._"

            if uc is not None:
                uc["_last_lifestyle_impact_text"] = answer
            return answer

        except Exception as e:
            logger.error(f"Error in get_lifestyle_glucose_impact: {e}")
            return {"error": f"Error computing lifestyle-glucose impact: {str(e)}"}