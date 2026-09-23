"""Shared helper: the date window for a patient's current program cycle.

Used by every tool that defaults an unspecified date range to 'this cycle'.
One definition so the logic can never drift between tools."""

from datetime import date
from dal.database import DatabaseManager


def cycle_window(patient_id: int):
    """(from_date, to_date) as 'YYYY-MM-DD' for the patient's current program
    cycle (active plan). If none is active, the most recent ALREADY-STARTED
    cycle. None if the patient has no started, dated plan (caller then keeps its
    own fallback). End is clamped to today, since an active plan's to_date is in
    the future and there's no data past today."""
    try:
        with DatabaseManager() as dm:
            plan = dm.get_current_active_plan(patient_id=patient_id)
            today = date.today().isoformat()
            if not (plan and plan.get("from_date")):
                plans = dm.get_user_plans(patient_id=patient_id, active_only=False)
                started = [p for p in plans
                           if p.get("from_date") and p["from_date"][:10] <= today]
                plan = started[0] if started else None
            if not (plan and plan.get("from_date")):
                return None
            start = plan["from_date"][:10]
            end = plan["to_date"][:10] if plan.get("to_date") else today
            if end > today:
                end = today
            return start, end
    except Exception:
        return None