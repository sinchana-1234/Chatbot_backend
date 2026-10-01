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
        "Get a patient's food log records. Filter by patient name or ID. "
        "For a single day, pass date_filter (YYYY-MM-DD). "
        "For a graph or any date range, pass from_date and to_date (YYYY-MM-DD), or a "
        "period phrase (e.g. 'this week', 'last month', 'this cycle') in `period` — the "
        "whole window is returned, not just the latest few. With no dates given, returns "
        "the latest `limit` records (default 10). "
        "Returns a list of food log entries with meal type, time, photo and macronutrient details."
    )

    def __init__(self):
        super().__init__()
        # Don't set user_context as instance variable to avoid Pydantic validation issues
    
    def set_user_context(self, user_context):
        """Set user context for role-based access control"""
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, 'user_context', user_context)

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
            date_filter: Optional[str] = None, from_date: Optional[str] = None,
            to_date: Optional[str] = None, period: Optional[str] = None,
            limit: int = 10) -> Dict[str, Any]:
        """
        Query food log records for a patient with role-based access control.
        Args:
            patient_id (int, optional): Patient ID.
            patient_name (str, optional): Patient name.
            date_filter (str, optional): Single date (YYYY-MM-DD) — one day's logs.
            from_date (str, optional): Range start (YYYY-MM-DD).
            to_date (str, optional): Range end (YYYY-MM-DD).
            period (str, optional): Period phrase ('this week', 'last month', 'this cycle').
            limit (int, optional): Max records when no date range is given.
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
            result = db_manager.get_foodlog(
                patient_id=patient_id,
                patient_name=patient_name,
                date_filter=date_obj,
                from_date=from_date,
                to_date=to_date,
                period=period,
                limit=limit,
            )

        # Stash a food-log graph payload for the frontend -- one marker per meal.
        # response_builder gates `_last_foodlog_chart_data` behind the chart keywords,
        # so "food log" gives the list and "food log graph" adds the graph.
        uc = getattr(self, "user_context", None)
        if uc is not None and isinstance(result, dict):
            meals = []
            for e in result.get("foodlog") or []:
                iso = e.get("actual_time")
                try:
                    dt = datetime.fromisoformat(iso) if iso else None
                except (ValueError, TypeError):
                    dt = None
                if not dt:
                    continue
                meals.append({
                    "meal_type": e.get("meal_type"),
                    # `date` lets the graph decide day vs week vs month; `time`
                    # places the marker within the day.
                    "date": dt.strftime("%Y-%m-%d"),
                    "time": dt.strftime("%I:%M %p"),
                    "photo_url": e.get("url"),
                    "calories": e.get("calories"),
                    "carbs_g": e.get("carbs_g"),
                    "protein_g": e.get("protein_g"),
                    "fat_g": e.get("fat_g"),
                })
            if meals:
                uc["_last_foodlog_chart_data"] = meals

                # Resolve the window label once (used by both branches).
                def _nice(d):
                    try:
                        return datetime.strptime(d, "%Y-%m-%d").strftime("%B %d, %Y")
                    except (ValueError, TypeError):
                        return d

                df, dto = result.get("date_from"), result.get("date_to")
                if df and dto and df != dto:
                    when = f" from {_nice(df)} to {_nice(dto)}"
                elif df:
                    when = f" for {_nice(df)}"
                elif date_obj:
                    when = f" for {date_obj.strftime('%B %d, %Y')}"
                else:
                    when = ""

                n = len(meals)
                q = (uc.get("_current_query") or "").lower()
                wants_graph = any(w in q for w in
                                  ("graph", "chart", "plot", "visual", "visualize", "diagram"))

                if wants_graph:
                    # Graph asked for: one-line caption; the graph carries each
                    # meal's photo + macros on hover.
                    uc["_last_foodlog_text"] = (
                        f"Here is the food log timeline{when}, showing {n} logged "
                        f"{'meal' if n == 1 else 'meals'}. Hover over any meal marker to "
                        f"view its photo and macronutrient breakdown."
                    )
                else:
                    # No graph: build the item list DETERMINISTICALLY, with a blank
                    # line between every element, so markdown never collapses it to
                    # one line (the LLM used single newlines -> rendered as spaces).
                    def _g(v):   # round macro grams/kcal to 1 dp, drop trailing .0
                        try:
                            f = round(float(v), 1)
                        except (TypeError, ValueError):
                            return v
                        return int(f) if f == int(f) else f

                    out = [f"Here is the food log{when}, showing {n} logged "
                           f"{'meal' if n == 1 else 'meals'}:", ""]
                    for i, e in enumerate(result.get("foodlog") or [], 1):
                        iso = e.get("actual_time")
                        try:
                            dt = datetime.fromisoformat(iso) if iso else None
                        except (ValueError, TypeError):
                            dt = None
                        mtype = (e.get("meal_type") or "Meal").title()
                        tstr = dt.strftime("%I:%M %p") if dt else ""
                        desc = e.get("description") or ""   # <-- confirm this field name
                        macros = []
                        if e.get("calories") is not None:
                            macros.append(f"Calories: {_g(e['calories'])} kcal")
                        if e.get("carbs_g") is not None:
                            macros.append(f"Carbs {_g(e['carbs_g'])} g")
                        if e.get("protein_g") is not None:
                            macros.append(f"Protein {_g(e['protein_g'])} g")
                        if e.get("fat_g") is not None:
                            macros.append(f"Fat {_g(e['fat_g'])} g")

                        header = f"**{i}. {mtype}"
                        if tstr:
                            header += f" — {tstr}"
                        header += "**"
                        out.append(header)
                        out.append("")
                        if desc:
                            out.append(f"**Description:** {desc}")
                            out.append("")
                        if macros:
                            out.append(" , ".join(macros))
                            out.append("")
                        if e.get("url"):
                            out.append(f"![{mtype}]({e['url']})")
                            out.append("")
                    uc["_last_foodlog_text"] = "\n".join(out).strip()

        return result


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