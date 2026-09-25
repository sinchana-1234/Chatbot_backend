#!/usr/bin/env python3
"""eHbA1c & TIR Summary Tool for Revival Medical System.

Calls the same metrics-api.glixify.ai/tirNehba1c endpoint the real doctor
dashboard uses, forwarding the doctor's own Cognito token. Returns eHbA1c/TIR
trend data (first-day-vs-last-day, 5-day periods, and full device cycles)."""

import os
import json
import logging
from typing import Optional
from datetime import date, timedelta
import httpx
from langchain.tools import BaseTool
from dal.postgres_db import resolve_range
from config import settings

logger = logging.getLogger(__name__)

EHBA1C_TIR_API_URL = os.getenv(
    "EHBA1C_TIR_API_URL",
    "https://metrics-api.glixify.ai/v2/data/chart/tirNehba1c"
)


def _cycle_window(patient_id: int):
    """The patient's current program cycle as (from_date, to_date), 'YYYY-MM-DD'.
    Active plan if any, else the most recent past plan. None if no dated plan."""
    try:
        from dal.database import DatabaseManager
        with DatabaseManager() as dm:
            plan = dm.get_current_active_plan(patient_id=patient_id)
            if not (plan and plan.get("from_date")):
                plans = dm.get_user_plans(patient_id=patient_id, active_only=False)
                # Never fall back onto a not-yet-started (future) plan — keep only
                # cycles that have already begun, newest first.
                today = date.today().isoformat()
                started = [p for p in plans if p.get("from_date") and p["from_date"][:10] <= today]
                plan = started[0] if started else None
            if not (plan and plan.get("from_date")):
                return None
            today = date.today().isoformat()
            start = plan["from_date"][:10]
            end = plan["to_date"][:10] if plan.get("to_date") else today
            if end > today:
                end = today
            return start, end
    except Exception:
        return None

def _fmt_pct(v, decimals=None):
    if v is None:
        return None
    return f"{float(v):.{decimals}f}" if decimals is not None else f"{float(v):g}"


def _fmt_period_range(start_iso, end_iso, fallback):
    """Format the actual data-covered period, e.g. 'September 9 – 24, 2026'."""
    from datetime import datetime as _dt
    try:
        d1 = _dt.fromisoformat(str(start_iso)[:19])
        d2 = _dt.fromisoformat(str(end_iso)[:19])
        if d1.year == d2.year and d1.month == d2.month:
            return f"{d1.strftime('%d-%m-%Y')} – {d2.strftime('%d-%m-%Y')}"
        if d1.year == d2.year:
            return f"{d1.strftime('%d-%m-%Y')} – {d2.strftime('%d-%m-%Y')}"
        return f"{d1.strftime('%d-%m-%Y')} – {d2.strftime('%d-%m-%Y')}"
    except Exception:
        return fallback


def _format_ehba1c_tir_prose(name, first_day, last_day, scope):
    """Deterministic first->last prose, delivered verbatim. States each metric's own
    direction factually — no global 'positive'/'improvement' verdict."""
    if not first_day or not last_day:
        return None
    name = name or "This patient"
    ft, lt = first_day.get("tir"), last_day.get("tir")
    fe, le = first_day.get("ehba1c"), last_day.get("ehba1c")

    clauses = []
    if ft is not None and lt is not None:
        if lt > ft:
            clauses.append(f"time in range improved from {_fmt_pct(ft)}% to {_fmt_pct(lt)}%")
        elif lt < ft:
            clauses.append(f"time in range dropped from {_fmt_pct(ft)}% to {_fmt_pct(lt)}%")
        else:
            clauses.append(f"time in range held at {_fmt_pct(ft)}%")
    if fe is not None and le is not None:
        if le > fe:
            clauses.append(f"estimated eHbA1c edged up from {_fmt_pct(fe, 2)}% to {_fmt_pct(le, 2)}%")
        elif le < fe:
            clauses.append(f"estimated eHbA1c eased down from {_fmt_pct(fe, 2)}% to {_fmt_pct(le, 2)}%")
        else:
            clauses.append(f"estimated eHbA1c held at {_fmt_pct(fe, 2)}%")

    if not clauses:
        return None
    period_str = _fmt_period_range(first_day.get("periodStart"), last_day.get("periodEnd"), scope)
    return f"{name}'s " + ", while ".join(clauses) + f".\n\n_Analyzed period: {period_str}_"

class EHbA1cTIRTool(BaseTool):
    """Fetches a patient's eHbA1c/TIR trend data (first day vs last day,
    5-day periods, and device cycles), using the same live dashboard endpoint."""
    name: str = "get_ehba1c_tir_trend"
    description: str = """Get a patient's eHbA1c and TIR trend over time — how glucose
    control has changed, comparing periods or CGM sensor cycles.

    Parameters:
    - patient_id (int): Patient ID (optional for patient role, required for staff queries)
    - patient_name (str): Patient name (alternative to patient_id for staff)
    - from_date (str): Start date (YYYY-MM-DD, YYYY-MM, or YYYY) — OPTIONAL
    - to_date (str): End date (YYYY-MM-DD, YYYY-MM, or YYYY) — OPTIONAL
    - period (str): OPTIONAL relative phrase, e.g. 'last 30 days', 'July 2026', 'this month'.
      If the user names NO period, omit from_date/to_date/period and the tool covers the
      patient's FULL available history — do NOT invent a default window.
    - specific_date (str): YYYY-MM-DD — set this if the user mentions a specific date, so
      the tool can find and report the 5-day period covering that date, instead of only
      the overall first-day-vs-last-day summary.

    Use this tool for queries like:
    - "How is patient X progressing?" / "compare this month with last month"
    - "eHbA1c trend for patient X" / "TIR history for patient X"
    - "eHbA1c trend on 2026-05-23" (specific_date="2026-05-23")
    - "how has patient X's glucose control changed over time"
    """

    def __init__(self):
        super().__init__()

    def set_user_context(self, user_context):
        object.__setattr__(self, 'user_context', user_context)

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
             from_date: Optional[str] = None, to_date: Optional[str] = None,
             specific_date: Optional[str] = None, period: Optional[str] = None) -> str:
        user_context = getattr(self, 'user_context', None)
        display_name = None

        if user_context and user_context.get('role_id') == 1:
            patient_id = user_context.get('user_id')
        elif not patient_id and patient_name:
            from dal.database import DatabaseManager
            with DatabaseManager() as db_manager:
                doctor_id = user_context.get('user_id') if user_context else None
                own_patients = db_manager.get_doctor_patients(doctor_user_id=doctor_id) if doctor_id else []
                own_matching = [
                    p for p in own_patients
                    if patient_name.lower() in f"{p.get('patient_first_name') or ''} {p.get('patient_last_name') or ''}".lower()
                ]
                if own_matching:
                    if len(own_matching) > 1:
                        return json.dumps({
                            "error": f"Multiple patients found matching '{patient_name}'",
                            "matching_patients": [
                                {"id": p["patient_id"], "name": f"{p.get('patient_first_name') or ''} {p.get('patient_last_name') or ''}".strip()}
                                for p in own_matching
                            ],
                            "suggestion": "Please specify which patient exactly."
                        })
                    patient_id = own_matching[0]["patient_id"]
                    display_name = (f"{own_matching[0].get('patient_first_name') or ''} "
                                    f"{own_matching[0].get('patient_last_name') or ''}".strip().title()) or None
                else:
                    users = db_manager.get_users()
                    matching = [
                        u for u in users
                        if patient_name.lower() in f"{u.first_name or ''} {u.last_name or ''}".lower()
                        and u.role_id == 1
                    ]
                    if not matching:
                        return json.dumps({"error": f"No patient found matching '{patient_name}'."})
                    if len(matching) > 1:
                        return json.dumps({
                            "error": f"Multiple patients found matching '{patient_name}'",
                            "matching_patients": [
                                {"id": u.id, "name": f"{u.first_name or ''} {u.last_name or ''}".strip()}
                                for u in matching
                            ],
                            "suggestion": "Please specify which patient exactly."
                        })
                    patient_id = matching[0].id
                    display_name = (f"{matching[0].first_name or ''} "
                                    f"{matching[0].last_name or ''}".strip().title()) or None
        elif not patient_id:
            return json.dumps({"error": "patient_id or patient_name is required for staff queries"})

        # Resolve the requested window. NO hidden 14-day default: nothing given →
        # all available history. The API is still called with concrete dates (it's
        # a from/to endpoint), but it returns the FULL device history regardless,
        # so the requested range is actually enforced by filtering the returned
        # 5-day periods below.
        start, end, period_label = resolve_range(from_date, to_date, period)
        range_requested = bool(start)
        if range_requested:
            from_date, to_date = start, end
        else:
            # No explicit range → default to the patient's current program cycle
            # (or most recent past cycle). Treat it as a requested range so the
            # returned 5-day periods are filtered to the cycle, not all history.
            cycle = _cycle_window(patient_id)
            if cycle:
                from_date, to_date = cycle
                range_requested = True
                period_label = f"{from_date} to {to_date}"
            else:
                to_date = date.today().isoformat()
                from_date = (date.today()
                             - timedelta(days=settings.TREND_ALL_HISTORY_LOOKBACK_DAYS)).isoformat()

        token = user_context.get('token') if user_context else None
        if not token:
            return json.dumps({"error": "No auth token available for this request"})

        url = f"{EHBA1C_TIR_API_URL.rstrip('/')}/{patient_id}"
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
                return json.dumps({"error": f"eHbA1c/TIR data unavailable: {data.get('statusMessage', 'unknown error')}"})

            content = data.get("content", {})
            summary = content.get("summary", [])       # first day vs last day (FULL device history)
            periods = content.get("periods", [])        # 5-day periods (FULL device history)
            device_cycles = content.get("deviceCycle", [])  # full CGM sensor cycles

            # The API ignores from_date/to_date and returns the full history, so
            # enforce the requested window HERE: keep only the 5-day periods that
            # overlap it. When no period was requested (all history), keep all.
            if range_requested:
                periods = [
                    p for p in periods
                    if p.get("periodStart") and p.get("periodEnd")
                    and p["periodStart"] <= to_date and p["periodEnd"] >= from_date
                ]

            # Rule 11: report an empty window honestly, never silently widen it.
            if range_requested and not periods:
                return json.dumps({"message": f"No eHbA1c/TIR data available for {period_label}."})
            if not summary and not periods:
                return json.dumps({"message": "No eHbA1c/TIR data available for this patient's available history."})

            first_day = next((s for s in summary if s.get("dayType") == "DAY_1"), None)
            last_day = next((s for s in summary if s.get("dayType") == "LAST_DAY"), None)
            # summary's DAY_1 / LAST_DAY are whole-device-history endpoints. When a
            # specific range (or the cycle default) is applied they are NOT this
            # window's endpoints. Use the FILTERED periods' own first and last period
            # as the comparison endpoints, so the reply compares WITHIN the window —
            # rather than nulling them, which left the model to invent numbers from
            # earlier in the chat (observed: a stale 62.2% TIR from a prior message).
            if range_requested:
                if periods:
                    _p0, _pN = periods[0], periods[-1]
                    first_day = {"periodStart": _p0.get("periodStart"),
                                 "tir": _p0.get("tir"), "ehba1c": _p0.get("ehba1c")}
                    last_day = {"periodEnd": _pN.get("periodEnd"),
                                "tir": _pN.get("tir"), "ehba1c": _pN.get("ehba1c")}
                else:
                    first_day = None
                    last_day = None

            # If the user asked about a specific date, find the 5-day period
            # that contains it — this is the finest granularity the API offers
            # (there's no true single-day breakdown besides first/last day).
            matched_period = None
            if specific_date:
                for p in periods:
                    period_start = p.get("periodStart")
                    period_end = p.get("periodEnd")
                    if period_start and period_end and period_start <= specific_date <= period_end:
                        matched_period = p
                        break

            # Stash the full period/cycle arrays for the frontend chart — kept
            # out of the LLM's own response (same pattern as AGP's time_blocks,
            # to avoid burning context or risking the "max iterations" issue).
            if user_context is not None:
                user_context['_last_ehba1c_tir_data'] = {
                    "periods": periods,
                    "device_cycles": device_cycles,
                    "first_day": first_day,
                    "last_day": last_day,
                }

            scope = period_label if range_requested else "all available data"
            if specific_date and not matched_period:
                message = (f"No period found covering {specific_date}; "
                           f"showing the {scope} trend instead.")
            elif matched_period:
                message = f"Data for the period covering {specific_date} is included below."
            else:
                message = f"eHbA1c/TIR trend for {scope} retrieved."

            # Deliver the trend as finished verbatim prose (same bypass as glucose/meals)
            # so the wording is deterministic and never overstated.
            if user_context is not None and not specific_date:
                prose = _format_ehba1c_tir_prose(display_name, first_day, last_day, scope)
                if prose:
                    user_context['_last_ehba1c_tir_text'] = prose

            return json.dumps({
                "patient_id": patient_id,
                "analyzed_period": scope,
                "first_day": first_day,
                "last_day": last_day,
                "requested_date": specific_date,
                "matched_period": matched_period,
                "period_count": len(periods),
                "cycle_count": len(device_cycles),
                "message": message
            })

        except httpx.HTTPStatusError as e:
            logger.error(f"eHbA1c/TIR API error: {e}")
            return json.dumps({"error": f"Failed to fetch eHbA1c/TIR data: {e.response.status_code}"})
        except Exception as e:
            logger.error(f"Error in EHbA1cTIRTool: {e}")
            return json.dumps({"error": f"eHbA1c/TIR trend error: {str(e)}"})

    async def _arun(self, patient_id=None, patient_name=None, from_date=None, to_date=None,
                    specific_date=None, period=None):
        return self._run(patient_id, patient_name, from_date, to_date, specific_date, period)