#!/usr/bin/env python3
"""AGP Chart Tool for Revival Medical System.

Calls the same metrics-api.glixify.ai/agp endpoint the real doctor dashboard
uses (forwarding the doctor's own Cognito token). Returns raw percentile
data (not an image) so the frontend can render an interactive chart."""

import os
import re
import json
import logging
from typing import Optional
from datetime import date, timedelta
import httpx
from langchain.tools import BaseTool
from dal.postgres_db import resolve_range
from dal.cycle_window import cycle_window
from config import settings

logger = logging.getLogger(__name__)

AGP_API_URL = os.getenv(
    "AGP_API_URL",
    "https://metrics-api.glixify.ai/v2/data/chart/agp"
)


def _parse_agp_data(response_json: dict) -> dict:
    """Same extraction logic as data_parser.py's parse_agp_data."""
    content = response_json.get("content", {})
    return {
        "summary": content.get("summary", {}),
        "tir": content.get("tir", {}),
        "agp_profile": content.get("agp", {}).get("time_blocks", []),
        "daily_metrics": content.get("daily_metrics", []),
    }


class AGPChartTool(BaseTool):
    """Fetches a patient's AGP (Ambulatory Glucose Profile) data for interactive
    charting on the frontend, using the same live dashboard endpoint as the real product."""
    name: str = "get_agp_chart"
    description: str = """Get a patient's AGP (Ambulatory Glucose Profile) chart and/or TIR summary.

    Parameters:
    - patient_id (int): Patient ID (optional for patient role, required for staff queries)
    - patient_name (str): Patient name (alternative to patient_id for staff)
    - from_date (str): Start date (YYYY-MM-DD, YYYY-MM, or YYYY) — OPTIONAL
    - to_date (str): End date (YYYY-MM-DD, YYYY-MM, or YYYY) — OPTIONAL
    - period (str): OPTIONAL relative phrase, e.g. 'last 30 days', 'July 2026', 'this month'.
      AGP is always computed over a BOUNDED window; if the user names NO period the tool
      uses a sensible default window — it does NOT span all history.
    - include_tir (bool): True if the user asked for Time in Range / TIR / a general overview.
    - include_agp (bool): True if the user asked for the AGP ribbon chart. Default true —
      set to FALSE only when the user asked for TIR specifically and did NOT mention AGP at all.

    Use this tool for queries like:
    - "Show me my AGP" (include_tir=false, include_agp=true)
    - "Time in range for patient 489" (include_tir=true, include_agp=false)
    - "AGP and TIR for patient X" / "TIR and AGP for X" (include_tir=true, include_agp=true)
    """

    def __init__(self):
        super().__init__()
   

    def set_user_context(self, user_context):
        object.__setattr__(self, 'user_context', user_context)

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
             from_date: Optional[str] = None, to_date: Optional[str] = None,
             include_tir: bool = False, include_agp: bool = True,
             period: Optional[str] = None) -> str:
        user_context = getattr(self, 'user_context', None)

        # Decide AGP vs TIR from the ACTUAL words in the request, not from the
        # flags the model guessed. "AGP" -> only AGP, "TIR" -> only TIR, and
        # asking for both -> both, every time. Only overrides when a keyword is
        # clearly present; otherwise the model's flags stand.
        q = ((user_context.get('_current_query') if user_context else '') or '').lower()
        wants_agp = bool(re.search(r'\bagp\b', q)) or ('glucose profile' in q) or ('ambulatory' in q)
        wants_tir = bool(re.search(r'\btir\b', q)) or ('time in range' in q) or ('time-in-range' in q)
        if wants_agp or wants_tir:
            include_agp = wants_agp
            include_tir = wants_tir

        if user_context and user_context.get('role_id') == 1:
            patient_id = user_context.get('user_id')
        elif not patient_id and patient_name:
            from dal.database import DatabaseManager
            with DatabaseManager() as db_manager:
                # Scope the search to THIS doctor's own patients first — resolves
                # most name ambiguity automatically (e.g. multiple "Vikas Reddy"
                # in the hospital, but only one assigned to this doctor).
                doctor_id = user_context.get('user_id') if user_context else None
                own_patients = db_manager.get_doctor_patients(doctor_user_id=doctor_id) if doctor_id else []
                matching = [
                    p for p in own_patients
                    if patient_name.lower() in f"{p.get('patient_first_name') or ''} {p.get('patient_last_name') or ''}".lower()
                ]
                matching_is_own_patients = bool(matching)

                # Fall back to searching all patients only if none of the doctor's
                # own patients match (e.g. covering-for-a-colleague scenarios).
                if not matching:
                    users = db_manager.get_users()
                    matching = [
                        u for u in users
                        if patient_name.lower() in f"{u.first_name or ''} {u.last_name or ''}".lower()
                        and u.role_id == 1
                    ]

            if not matching:
                return json.dumps({"error": f"No patient found matching '{patient_name}'."})
            if len(matching) > 1:
                if matching_is_own_patients:
                    names = [
                        {"id": p["patient_id"], "name": f"{p.get('patient_first_name') or ''} {p.get('patient_last_name') or ''}".strip()}
                        for p in matching
                    ]
                else:
                    names = [
                        {"id": u.id, "name": f"{u.first_name or ''} {u.last_name or ''}".strip()}
                        for u in matching
                    ]
                return json.dumps({
                    "error": f"Multiple patients found matching '{patient_name}'",
                    "matching_patients": names,
                    "suggestion": "Please specify which patient exactly."
                })
            patient_id = matching[0]["patient_id"] if matching_is_own_patients else matching[0].id
        elif not patient_id:
            return json.dumps({"error": "patient_id or patient_name is required for staff queries"})

        # AGP is a clinical report defined over a BOUNDED window, so — unlike the
        # trend/correlation tools — it never analyses "all history". If the user
        # named a period, use it; otherwise fall back to the configured default
        # window (AGP is undefined without one). DELIBERATE exception to the
        # "no period -> all history" rule: do NOT change it to match the trend tools.
        start, end, _ = resolve_range(from_date, to_date, period)
        if start:
            from_date, to_date = start, end
        else:
            cw = cycle_window(patient_id)
            if cw:
                from_date, to_date = cw
            else:
                to_date = date.today().isoformat()
                from_date = (date.today()
                             - timedelta(days=settings.AGP_DEFAULT_WINDOW_DAYS)).isoformat()

        token = user_context.get('token') if user_context else None
        if not token:
            return json.dumps({"error": "No auth token available for this request"})

        url = f"{AGP_API_URL.rstrip('/')}/{patient_id}"
        params = {"from_date": from_date, "to_date": to_date}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "loginuserid": str(user_context.get('user_id')),
            "loginroleid": str(user_context.get('role_id')),
            "x-timezone": "Asia/Kolkata",
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            if data.get("status") != 200:
                return json.dumps({"error": f"AGP data unavailable: {data.get('statusMessage', 'unknown error')}"})

            agp_data = _parse_agp_data(data)
            time_blocks = agp_data["agp_profile"]
            if not time_blocks:
                return json.dumps({"message": f"No AGP data found for patient {patient_id} in this date range."})

            # Stash raw chart data where chat_routes.py can retrieve it directly —
            # never put large structured data in what the LLM has to read, or it
            # burns context/can cause the agent to loop (observed previously with
            # a base64 image; same risk applies to a large time_blocks array).
            if user_context is not None:
                user_context['_last_agp_chart_data'] = {
                    "time_blocks": None if (include_tir and not include_agp) else time_blocks,
                    "daily_metrics": agp_data["daily_metrics"],
                    "summary": agp_data["summary"],
                    "tir": agp_data["tir"] if include_tir else None,
                }

            actual_period = agp_data["summary"].get("Monitoring period", {}).get("Results", "unknown period")
            # time_blocks are deliberately NOT returned to the LLM. They are large
            # (12 blocks × 5 percentiles) and, once in the model's context, it dumps
            # the whole table into chat even though the chart already shows every
            # value. The chart receives them via _last_agp_chart_data above; the LLM
            # only needs the headline summary to write its short reply.
            return json.dumps({
                "patient_id": patient_id,
                "requested_from_date": from_date,
                "requested_to_date": to_date,
                "actual_monitoring_period": actual_period,
                "summary": agp_data["summary"],
                "tir": agp_data["tir"] if include_tir else None,
                "message": f"Data available for: {actual_period}. "
                            f"Use THIS exact period in your reply, not the requested "
                            f"range. Report ONLY the short summary metrics (eHbA1c, "
                            f"mean glucose, CV, and TIR if provided). The chart already "
                            f"shows the per-time-of-day values — do NOT list or table "
                            f"any 00:00 / 02:00 / … percentile values."
            })

        except httpx.HTTPStatusError as e:
            logger.error(f"AGP API error: {e}")
            return json.dumps({"error": f"Failed to fetch AGP data: {e.response.status_code}"})
        except Exception as e:
            logger.error(f"Error in AGPChartTool: {e}")
            return json.dumps({"error": f"Chart generation error: {str(e)}"})

    async def _arun(self, patient_id=None, patient_name=None, from_date=None, to_date=None,
                    include_tir=False, include_agp=True, period=None):
        return self._run(patient_id, patient_name, from_date, to_date, include_tir, include_agp, period)