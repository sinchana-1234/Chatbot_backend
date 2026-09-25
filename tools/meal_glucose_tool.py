"""
Meal -> glucose impact correlation.

For each logged meal (foodlog.actual_time, Postgres) this measures the glucose
response in the standard post-meal window (glucose_readings.local_event_time,
Postgres) and ranks meals by how much glucose rose after them. Meals and glucose
now live in the same database on the same patient-local time basis, so this is a
clean same-DB correlation.

The answer is built entirely in Python and delivered verbatim (chat_routes returns
it directly, bypassing the LLM's formatting) so the wording is deterministic.

Honesty is built in: if a patient has no meals, or meals but no glucose in the
post-meal windows, or only a couple of usable meals, the output says exactly that
rather than inventing a ranking.
"""

import json
import logging
from collections import defaultdict
from typing import Optional, Dict, Any

from langchain.tools import BaseTool
from sqlalchemy import text

from dal.postgres_db import SessionLocalPG, resolve_range, date_where
from dal.database import DatabaseManager
from dal.cycle_window import cycle_window
from config import settings

logger = logging.getLogger(__name__)


_MEAL_TYPE_NORMALIZE = {
    "break fast": "Breakfast",
    "breakfast": "Breakfast",
    "lunch": "Lunch",
    "dinner": "Dinner",
    "snack": "Snack",
    "other": "Other",
}


def _norm_meal_type(mt: Optional[str]) -> str:
    if not mt:
        return "Meal"
    return _MEAL_TYPE_NORMALIZE.get(mt.strip().lower(), mt.strip().title())


def _carbs_from_analysis(analysis_data: Optional[str]) -> Optional[int]:
    """Pull carbohydrate grams out of the analysis_data JSON, if present.
    Real format nests it under macronutrients.carbohydrates.grams; 'null' or
    image-only rows have no carbs."""
    if not analysis_data or str(analysis_data).strip().lower() in ("null", ""):
        return None
    try:
        d = json.loads(analysis_data)
        carbs = (d.get("macronutrients") or {}).get("carbohydrates")
        if isinstance(carbs, dict):          # real format: {"grams": 70, ...}
            c = carbs.get("grams")
        elif carbs is not None:              # flat number, if ever stored that way
            c = carbs
        else:                                # legacy top-level fallback
            c = d.get("carbohydrates")
        return round(float(c)) if c is not None else None
    except (ValueError, TypeError, AttributeError):
        return None


def _format_meal_impact(patient_name: Optional[str], meals: list) -> str:
    """meals: list of dicts with keys meal_type, date_str, carbs, baseline, peak, pts, rise."""
    name = patient_name or "This patient"

    if not meals:
        return f"No meals are logged for {name}, so meal impact on glucose can't be assessed."

    measured = [m for m in meals if m.get("rise") is not None]
    if not measured:
        n = len(meals)
        subj = "meal is" if n == 1 else "meals are"
        after = "it" if n == 1 else "those meals"
        poss = "its" if n == 1 else "their"
        return (
            f"There isn't enough matching data to determine which meals affect {name}'s glucose.\n\n"
            f"{n} {subj} logged, but there are no glucose readings within {settings.MEAL_POSTMEAL_WINDOW_HOURS} hours "
            f"after {after}, so {poss} impact cannot be assessed.\n"
            f"More meal and CGM data from the same period is needed to identify meal-related glucose changes."
        )

    groups = defaultdict(list)
    for m in measured:
        groups[m["meal_type"]].append(m)

    stats = []
    for mt, items in groups.items():
        avg = round(sum(i["rise"] for i in items) / len(items))
        top = max(items, key=lambda i: i["rise"])
        stats.append({"type": mt, "n": len(items), "avg": avg, "top": top})
    stats.sort(key=lambda s: s["avg"], reverse=True)

    top = stats[0]
    t = top["top"]
    dish = (t.get("description") or "").strip() or top["type"].lower()
    carbs = f" ({t['carbs']} g carbs)" if t.get("carbs") is not None else ""

    lines = [
        f"**Meal Impact on Glucose \u2014 {name}**",
        "",
        f"{top['type']} had the biggest effect, raising glucose by an average of "
        f"**+{top['avg']} mg/dL** across {top['n']} meals.",
        "",
        "**Biggest single spike:**",
        f"{dish}{carbs} \u2014 {t['date_str']}, glucose rose "
        f"**{t['baseline']} \u2192 {t['peak']} mg/dL**.",
        "",
        "**Average rise by meal type:**",
    ]
    for s in stats:
        lines.append(f"- **{s['type']}** \u2014 +{s['avg']} mg/dL")

    return "\n".join(lines)


def _correlate(patient_id: int, start=None, end=None) -> list:
    """Run the single-query correlation in Postgres and return per-meal dicts.
    Meals are filtered to [start, end] on their logged time; None/None = all history.
    (Each meal's glucose window stays anchored to that meal's own timestamp.)"""
    win = f"INTERVAL '{settings.MEAL_POSTMEAL_WINDOW_HOURS} hours'"
    tol = f"INTERVAL '{settings.MEAL_BASELINE_TOLERANCE_MIN} minutes'"
    meal_where, meal_params = date_where("f.actual_time", start, end, prefix="m")
    sql = text(f"""
        SELECT f.id, f.actual_time, f.meal_type, f.description, f.analysis_data,
               w.pts, w.peak, b.baseline
        FROM foodlog f
        LEFT JOIN LATERAL (
            SELECT COUNT(*) AS pts, MAX(g.glucose_value) AS peak
            FROM glucose_readings g
            WHERE g.patient_id = f.patient_id
              AND g.local_event_time >= f.actual_time
              AND g.local_event_time < f.actual_time + {win}
        ) w ON true
        LEFT JOIN LATERAL (
            SELECT g.glucose_value AS baseline
            FROM glucose_readings g
            WHERE g.patient_id = f.patient_id
              AND g.local_event_time BETWEEN f.actual_time - {tol} AND f.actual_time + {tol}
            ORDER BY ABS(EXTRACT(EPOCH FROM (g.local_event_time - f.actual_time)))
            LIMIT 1
        ) b ON true
        WHERE f.patient_id = :pid AND f.status = 1 AND f.actual_time IS NOT NULL AND {meal_where}
        ORDER BY f.actual_time
    """)
    pg = SessionLocalPG()
    try:
        rows = pg.execute(sql, {"pid": patient_id, **meal_params}).mappings().all()
    finally:
        pg.close()

    meals = []
    for r in rows:
        baseline = r["baseline"]
        peak = r["peak"]
        pts = r["pts"] or 0
        rise = round(peak - baseline) if (pts > 0 and baseline is not None and peak is not None) else None
        meals.append({
            "meal_type": _norm_meal_type(r["meal_type"]),
            "description": r["description"],
            "date_str": (f"{r['actual_time'].strftime('%B')} {r['actual_time'].day}, {r['actual_time'].year}"
                         if r["actual_time"] else "unknown date"),
            "carbs": _carbs_from_analysis(r["analysis_data"]),
            "baseline": round(baseline) if baseline is not None else None,
            "peak": round(peak) if peak is not None else None,
            "pts": pts,
            "rise": rise,
        })
    return meals


class MealGlucoseImpactTool(BaseTool):
    """Correlate a patient's logged meals with their post-meal glucose response."""

    name: str = "get_meal_glucose_impact"
    description: str = (
        "Answer 'which meals affect this patient's glucose most?' / 'how do this patient's "
        "meals affect their glucose?'. For each logged meal it measures the glucose rise in "
        "the 2 hours after the meal and ranks meals by that rise, including carbs when "
        "available. Requires a patient name or ID. Optional from_date/to_date ('YYYY-MM-DD', "
        "'YYYY-MM' or 'YYYY') or a period phrase (e.g. 'last 30 days', 'July 2026') narrow the "
        "window; omit all three for full history. Returns a finished, ready-to-send summary "
        "(relay it verbatim, do not reformat)."
    )

    def __init__(self):
        super().__init__()

    def set_user_context(self, user_context):
        object.__setattr__(self, "user_context", user_context)

    def _resolve_patient(self, patient_id, patient_name):
        """Return (patient_id, display_name). Doctor-scoped for staff; own id for patients."""
        uc = getattr(self, "user_context", None)

        # Patient role: force their own id
        if uc and uc.get("role_id") == 1:
            patient_id = uc.get("user_id")

        with DatabaseManager() as dm:
            # Resolve by name (staff) if no id yet, scoped to the doctor's own patients
            if patient_id is None and patient_name:
                doctor_id = uc.get("user_id") if uc else None
                patients = dm.get_doctor_patients(doctor_user_id=doctor_id, active_only=True) if doctor_id else []
                name_l = patient_name.lower().strip()
                for p in patients:
                    full = f"{p.get('patient_first_name', '')} {p.get('patient_last_name', '')}".lower().strip()
                    if name_l in full:
                        return p["patient_id"], full.title() or None
                return None, None

            # Look up display name for a known id
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
            meals = _correlate(patient_id, start, end)
            answer = _format_meal_impact(display_name, meals)
            answer += f"\n\n_Period analyzed: {label}._"

            # Deliver verbatim via the same side-channel the other tools use.
            if uc is not None:
                uc["_last_meal_impact_text"] = answer
            return answer

        except Exception as e:
            logger.error(f"Error in get_meal_glucose_impact: {e}")
            return {"error": f"Error computing meal-glucose impact: {str(e)}"}