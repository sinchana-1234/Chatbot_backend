from typing import Optional, Dict, Any
from datetime import datetime
from langchain.tools import BaseTool
from dal.database import DatabaseManager

class FoodlogTool(BaseTool):
    """
    Tool for querying food log records for a patient. Returns the latest food log entries for a patient by name or ID, with optional date filtering and result limit.
    """
    name: str = "get_foodlog"
    description: str = (
        "Get the latest food log records for a patient. "
        "You can filter by patient name or ID, set a date filter (YYYY-MM-DD), and limit the number of results (default 10). "
        "Returns a list of food log entries with type, description, activity date, and other details."
    )

    def __init__(self):
        super().__init__()
        # Don't set user_context as instance variable to avoid Pydantic validation issues
    
    def set_user_context(self, user_context):
        """Set user context for role-based access control"""
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, 'user_context', user_context)

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
            date_filter: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """
        Query food log records for a patient with role-based access control.
        Args:
            patient_id (int, optional): Patient ID.
            patient_name (str, optional): Patient name.
            date_filter (str, optional): Date filter in YYYY-MM-DD format.
            limit (int, optional): Max number of records to return.
        Returns:
            dict: Food log records and metadata.
        """
        # Enforce role-based access control
        user_context = getattr(self, "user_context", None)
        if user_context and user_context.get('role_id') == 1:  # Patient role
            # Patients can only access their own food logs
            patient_id = user_context.get('user_id')
            patient_name = None  # Override any patient_name to enforce access control
        elif patient_id is None and patient_name is None:
            # For medical staff, if no patient specified, this might be an error
            return {"error": "Please specify a patient ID or patient name for the food log query."}
        
        date_obj = None
        if date_filter:
            try:
                date_obj = datetime.strptime(date_filter, "%Y-%m-%d")
            except Exception:
                return {"error": "Invalid date_filter format. Use YYYY-MM-DD."}
        with DatabaseManager() as db_manager:
            return db_manager.get_foodlog(
                patient_id=patient_id,
                patient_name=patient_name,
                date_filter=date_obj,
                limit=limit
        )


class FoodlogUploadersTool(BaseTool):
    """
    Tool for listing which patients uploaded a food log on a specific date,
    scoped to the logged-in staff member's own patients (or all patients for
    roles with system-wide access). Sibling of FoodlogTool: that one looks up
    one named patient's records, this one answers roster-wide "who uploaded"
    questions.
    """
    name: str = "get_foodlog_uploaders_by_date"
    description: str = (
        "Get the list of patients who uploaded a food log on a specific date. "
        "Use this for aggregate/roster-wide questions like "
        "'who uploaded a food log today', 'who uploaded yesterday', "
        "or 'who uploaded a food log on 2026-09-15' — as opposed to get_foodlog, "
        "which looks up records for one specific named patient. "
        "Always resolve relative phrases like 'today' or 'yesterday' to a concrete "
        "date in YYYY-MM-DD format (using the current date given in your instructions) "
        "before calling this tool — this tool only accepts an exact date, not phrases. "
        "Set include_items=True when the question also asks what each patient ate "
        "(e.g. 'who uploaded today and what did they eat', 'show what everyone logged "
        "on 2026-09-15') — this returns each patient's actual food items alongside their "
        "name. Leave include_items=False (default) for a plain name-only list. "
        "This tool is restricted to medical staff; it is not available to patients."
    )

    def __init__(self):
        super().__init__()
        # Don't set user_context as instance variable to avoid Pydantic validation issues

    def set_user_context(self, user_context):
        """Set user context for role-based access control"""
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, 'user_context', user_context)

    def _run(self, date: str, include_items: bool = False) -> Dict[str, Any]:
        """
        List patients who uploaded a food log on the given date.
        Args:
            date (str): The date to check, in YYYY-MM-DD format.
            include_items (bool): If True, also return each patient's food
                items for that date, not just their name.
        Returns:
            dict: date, count, and the list of uploading patients.
        """
        user_context = getattr(self, "user_context", None)

        # Enforce role-based access control: patients cannot see other
        # patients' upload activity, so this tool is staff-only. (It is not
        # even registered for the patient role, but guard here too in case
        # user_context is ever missing or stale.)
        if user_context and user_context.get('role_id') == 1:  # Patient role
            return {"error": "Access denied: this query is only available to medical staff."}

        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
        except Exception:
            return {"error": "Invalid date format. Use YYYY-MM-DD."}

        with DatabaseManager() as db_manager:
            # Determine roster scope: a specific doctor's own patients, or
            # every active patient system-wide for roles that can see everyone.
            roster_patient_ids = None
            if user_context and not user_context.get('can_access_all_patients'):
                doctor_id = user_context.get('user_id')
                doctor_patients = db_manager.get_doctor_patients(doctor_user_id=doctor_id, active_only=True)
                roster_patient_ids = [p['patient_id'] for p in doctor_patients]
            elif user_context and user_context.get('can_access_all_patients'):
                roster_patient_ids = db_manager.get_all_active_patient_ids()
            # If there's no user_context at all, fall back to system-wide
            # (matches the permissive default used elsewhere when context is missing)

            return db_manager.get_foodlog_uploaders_by_date(
                date_filter=date_obj,
                roster_patient_ids=roster_patient_ids,
                include_items=include_items
            )