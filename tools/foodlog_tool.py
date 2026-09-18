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
