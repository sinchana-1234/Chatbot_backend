#!/usr/bin/env python3
"""
Health Progress Tools
Fetches daily vitals (glucose, TIR, sleep, activity, heart rate, stress/HRV)
for a patient from the healthProgress endpoint.

ARCHITECTURE NOTE: this used to be a single tool with a 'metric' parameter
the LLM had to remember to set. That failed repeatedly in practice -- the
LLM would correctly pick this tool but then silently drop 'metric' and fall
through to the glucose default, even with explicit prompt instructions
telling it not to. Tool NAME selection is a much stronger signal than
parameter-filling, so this is now split into one tool per metric, each with
its own clear name/description and NO metric parameter to forget. All six
share the same fetch/chart/summarize logic via HealthProgressBase.

The base URL is read from HEALTH_PROGRESS_BASE_URL in .env (loaded into
os.environ by load_dotenv() at app startup) so these tools point at
whichever environment (QA / production) the rest of the backend is
configured for, without needing a code change to switch. No fallback
default is used -- if the setting is missing, the app fails loudly at
import time rather than silently querying the wrong environment.

Each tool stashes a chart-ready payload on `self.last_chart_data` -- a
side-channel the chat API route reads after the agent finishes, since a
LangChain tool can only return text to the LLM itself. `last_chart_data`
is cleared after each read so a stale chart never leaks into an unrelated
later answer.
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Optional
import httpx
from langchain.tools import BaseTool
from dal.database import DatabaseManager
from dal.postgres_db import resolve_range
from config import settings

logger = logging.getLogger(__name__)

HEALTH_PROGRESS_BASE = os.getenv("HEALTH_PROGRESS_BASE_URL")
if not HEALTH_PROGRESS_BASE:
    raise RuntimeError(
        "HEALTH_PROGRESS_BASE_URL is not set in .env — HealthProgressTool cannot function without it."
    )


def _format_glucose_trend_prose(summary: dict, from_date: str, to_date: str,
                                is_fallback: bool = False) -> str:
    """Build the glucose-trend answer as a finished bulleted summary, delivered to
    the user verbatim (chat_routes returns it directly, bypassing the LLM's
    formatting). Every number comes straight from glucoseDailyAnalytics fields
    (meanGlucose / minGlucose / maxGlucose / tirPercent / tarPercent / tbrPercent /
    glycemicVariabilityCv) -- nothing is invented or re-derived by the model."""
    # Human-friendly range, e.g. "August 26 to September 1, 2026". Uses .day/.year
    # ints rather than %-d/%#d so it is cross-platform (Windows-safe).
    try:
        d1 = datetime.strptime(from_date, "%Y-%m-%d")
        d2 = datetime.strptime(to_date, "%Y-%m-%d")
        if d1.year == d2.year:
            date_range = f"{d1.strftime('%B')} {d1.day} to {d2.strftime('%B')} {d2.day}, {d2.year}"
        else:
            date_range = f"{d1.strftime('%B')} {d1.day}, {d1.year} to {d2.strftime('%B')} {d2.day}, {d2.year}"
    except (ValueError, TypeError):
        date_range = f"{from_date} to {to_date}"

    avg = summary.get("avg_glucose")
    lo  = summary.get("min_glucose")
    hi  = summary.get("max_glucose")
    tir = summary.get("avg_tir_pct")
    tar = summary.get("avg_tar_pct")
    tbr = summary.get("avg_tbr_pct")
    cv  = summary.get("avg_cv_pct")

    if is_fallback:
        header = ("No recent glucose data is available. Here are the most recent "
                  f"available glucose trends from {date_range}:")
    else:
        header = f"Here are the glucose trends from {date_range}:"

    lines = [f"* **Average glucose:** {avg} mg/dL"]
    if lo is not None and hi is not None:
        # \u2013 is an en dash (kept as an escape so the source stays ASCII).
        lines.append(f"* **Glucose range:** {lo}\u2013{hi} mg/dL")
    if tir is not None:
        lines.append(f"* **Time in Range (TIR):** {tir}%")
    if tbr is not None:
        lines.append(f"* **Below range:** {tbr}%")
    if tar is not None:
        lines.append(f"* **Above range:** {tar}%")
    if cv is not None:
        lines.append(f"* **Glucose variability (CV):** {cv}%")

    return header + "\n\n" + "\n".join(lines)


class HealthProgressBase(BaseTool):
    """Shared logic for all health-progress tools. Never instantiated
    directly -- only the leaf subclasses below (one per metric) are."""

    def set_user_context(self, user_context):
        object.__setattr__(self, 'user_context', user_context)
        object.__setattr__(self, 'last_chart_data', None)

    def _fetch(self, patient_id: int, from_date: str, to_date: str, auth_token: str) -> dict:
        resp = httpx.get(
            f"{HEALTH_PROGRESS_BASE}/{patient_id}/{from_date}/{to_date}",
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=15.0
        )
        resp.raise_for_status()
        return resp.json()

    def _resolve_patient_id(self, patient_id: Optional[int], patient_name: Optional[str]) -> tuple:
        if patient_id:
            return patient_id, None
        if not patient_name:
            return None, json.dumps({
                "status": "not_found",
                "notice": "Either patient_id or patient_name is required."
            })

        user_context = getattr(self, 'user_context', None)

        with DatabaseManager() as db_manager:
            doctor_id = user_context.get('user_id') if user_context else None
            own_patients = db_manager.get_doctor_patients(doctor_user_id=doctor_id) if doctor_id else []
            own_matching = [
                p for p in own_patients
                if patient_name.lower() in f"{p.get('patient_first_name') or ''} {p.get('patient_last_name') or ''}".lower()
            ]

            if own_matching:
                if len(own_matching) > 1:
                    return None, json.dumps({
                        "status": "ambiguous_name",
                        "notice": f"Multiple patients match '{patient_name}'. Ask the user which one.",
                        "matching_patients": [
                            {
                                "id": p["patient_id"],
                                "name": f"{p.get('patient_first_name') or ''} {p.get('patient_last_name') or ''}".strip(),
                                "email": p.get("patient_email")
                            } for p in own_matching
                        ]
                    })
                return own_matching[0]["patient_id"], None

            users = db_manager.get_users()
            matching_users = [
                u for u in users
                if patient_name.lower() in f"{u.first_name or ''} {u.last_name or ''}".lower()
                and u.role_id == 1
            ]

            if not matching_users:
                return None, json.dumps({
                    "status": "not_found",
                    "notice": f"No patient found with name containing '{patient_name}'."
                })

            if len(matching_users) > 1:
                return None, json.dumps({
                    "status": "ambiguous_name",
                    "notice": f"Multiple patients match '{patient_name}'. Ask the user which one.",
                    "matching_patients": [
                        {
                            "id": u.id,
                            "name": f"{u.first_name or ''} {u.last_name or ''}".strip(),
                            "email": u.email
                        } for u in matching_users
                    ]
                })

            return matching_users[0].id, None

    def _build_chart(self, metric: str, glucose_daily: list, data: dict, from_date: str, to_date: str) -> dict:
        if metric == "hba1c":
            valid = [d for d in glucose_daily if d.get("estimatedHba1c") is not None]
            if not valid:
                return None
            return {
                "type": "area",
                "title": f"eHbA1c Trend - {from_date} to {to_date}",
                "x_labels": [d["glucoseDate"] for d in valid],
                "series": [{"name": "eHbA1c (%)", "values": [d["estimatedHba1c"] for d in valid]}],
                "color": "#4a90e2",
                "stats": {"avg_tir_pct": None, "avg_bp": None, "avg_hba1c": None}
            }

        if metric == "fbs":
            valid = [d for d in glucose_daily if d.get("fbs") is not None]
            if not valid:
                return None
            return {
                "type": "area",
                "title": f"Fasting Blood Sugar - {from_date} to {to_date}",
                "x_labels": [d["glucoseDate"] for d in valid],
                "series": [{"name": "Fasting BS (mg/dL)", "values": [d["fbs"] for d in valid]}],
                "color": "#e0576a",
                "stats": {"avg_tir_pct": None, "avg_bp": None, "avg_hba1c": None}
            }

        if metric == "tir":
            return {
                "type": "overlayArea",
                "title": f"Time in Range - {from_date} to {to_date}",
                "x_labels": [d["glucoseDate"] for d in glucose_daily],
                "series": [
                    {"name": "Above (%)", "values": [d["tarPercent"] for d in glucose_daily]},
                    {"name": "In Range (%)", "values": [d["tirPercent"] for d in glucose_daily]},
                    {"name": "Below (%)", "values": [d["tbrPercent"] for d in glucose_daily]},
                ],
                "stats": {
                    "avg_tir_pct": round(sum(d["tirPercent"] for d in glucose_daily) / len(glucose_daily), 1),
                    "avg_bp": None, "avg_hba1c": None,
                }
            }

        if metric == "sleep":
            sleep = data.get("sleep") or []
            if not sleep:
                return None
            return {
                "type": "line",
                "title": f"Sleep Quality - {from_date} to {to_date}",
                "x_labels": [d["readingDate"] for d in sleep],
                "series": [
                    {"name": "Light Sleep (hrs)", "values": [round(d.get("lightSleep", 0) / 60, 1) for d in sleep]},
                    {"name": "Deep Sleep (hrs)", "values": [round(d.get("deepSleep", 0) / 60, 1) for d in sleep]},
                    {"name": "REM Sleep (hrs)", "values": [round(d.get("remSleep", 0) / 60, 1) for d in sleep]},
                ],
                "stats": {"avg_tir_pct": None, "avg_bp": None, "avg_hba1c": None}
            }

        if metric == "activity":
            activity = data.get("activity") or []
            if not activity:
                return None
            return {
                "type": "line",
                "title": f"Activity - {from_date} to {to_date}",
                "x_labels": [d["readingDate"] for d in activity],
                "series": [
                    {"name": "Steps", "values": [d.get("totalSteps", 0) for d in activity]},
                    {"name": "Calories (kcal)", "values": [d.get("calories", 0) for d in activity]},
                ],
                "dual_axis": True,
                "stats": {"avg_tir_pct": None, "avg_bp": None, "avg_hba1c": None}
            }

        if metric == "bp":
            bp = data.get("bp") or []
            if not bp:
                return None
            return {
                "type": "line",
                "title": f"Blood Pressure - {from_date} to {to_date}",
                "x_labels": [d["readingDate"] for d in bp],
                "series": [
                    {"name": "Systolic (mmHg)", "values": [d.get("averageSystolic", 0) for d in bp]},
                    {"name": "Diastolic (mmHg)", "values": [d.get("averageDiastolic", 0) for d in bp]},
                ],
                "stats": {"avg_tir_pct": None, "avg_bp": None, "avg_hba1c": None}
            }

        if metric == "heart_rate":
            hr = data.get("hr") or []
            if not hr:
                return None
            return {
                "type": "line",
                "title": f"Heart Rate - {from_date} to {to_date}",
                "x_labels": [d["readingDate"] for d in hr],
                "series": [
                    {"name": "Average (bpm)", "values": [d.get("averageHeartRate", 0) for d in hr]},
                    {"name": "Low (bpm)", "values": [d.get("lowHeartRate", 0) for d in hr]},
                    {"name": "High (bpm)", "values": [d.get("highHeartRate", 0) for d in hr]},
                    {"name": "Resting (bpm)", "values": [d.get("restingHeartRate", 0) for d in hr]},
                ],
                "stats": {"avg_tir_pct": None, "avg_bp": None, "avg_hba1c": None}
            }

        if metric == "stress_hrv":
            stress = data.get("stress") or []
            hrv = data.get("hrv") or []
            if not stress and not hrv:
                return None
            dates = [d["readingDate"] for d in stress] if stress else [d["readingDate"] for d in hrv]
            hrv_by_date = {d["readingDate"]: d.get("average", 0) for d in hrv}
            stress_by_date = {d["readingDate"]: d.get("average", 0) for d in stress}
            return {
                "type": "line",
                "title": f"Stress / HRV - {from_date} to {to_date}",
                "x_labels": dates,
                "series": [
                    {"name": "Stress (%)", "values": [stress_by_date.get(dt, 0) for dt in dates]},
                    {"name": "HRV (ms)", "values": [hrv_by_date.get(dt, 0) for dt in dates]},
                ],
                "stats": {"avg_tir_pct": None, "avg_bp": None, "avg_hba1c": None}
            }

        chart_data = {
            "type": "line",
            "title": f"Mean glucose - {from_date} to {to_date}",
            "x_labels": [d["glucoseDate"] for d in glucose_daily],
            "series": [{"name": "Mean glucose (mg/dL)", "values": [d["meanGlucose"] for d in glucose_daily]}],
            "stats": {
                "avg_tir_pct": round(sum(d["tirPercent"] for d in glucose_daily) / len(glucose_daily), 1),
                "avg_bp": None,
                "avg_hba1c": round(
                    sum(d["estimatedHba1c"] for d in glucose_daily if d.get("estimatedHba1c")) /
                    max(1, len([d for d in glucose_daily if d.get("estimatedHba1c")])), 1
                )
            }
        }
        bp = data.get("bp") or []
        if bp:
            avg_sys = round(sum(d["averageSystolic"] for d in bp) / len(bp))
            avg_dia = round(sum(d["averageDiastolic"] for d in bp) / len(bp))
            chart_data["stats"]["avg_bp"] = f"{avg_sys}/{avg_dia}"
        return chart_data

    def _summarize(self, metric: str, glucose_daily: list, data: dict) -> dict:
        if metric == "glucose":
            valid = [d for d in glucose_daily if d.get("meanGlucose") is not None]
            if not valid:
                return {"days_count": 0}
            means = [d["meanGlucose"] for d in valid]
            mins  = [d["minGlucose"] for d in valid if d.get("minGlucose") is not None]
            maxs  = [d["maxGlucose"] for d in valid if d.get("maxGlucose") is not None]
            tirs  = [d["tirPercent"] for d in valid if d.get("tirPercent") is not None]
            tars  = [d["tarPercent"] for d in valid if d.get("tarPercent") is not None]
            tbrs  = [d["tbrPercent"] for d in valid if d.get("tbrPercent") is not None]
            cvs   = [d["glycemicVariabilityCv"] for d in valid if d.get("glycemicVariabilityCv") is not None]
            return {
                "days_count": len(valid),
                "avg_glucose": round(sum(means) / len(means)),
                "min_glucose": round(min(mins)) if mins else None,
                "max_glucose": round(max(maxs)) if maxs else None,
                "avg_tir_pct": round(sum(tirs) / len(tirs), 1) if tirs else None,
                "avg_tar_pct": round(sum(tars) / len(tars), 1) if tars else None,
                "avg_tbr_pct": round(sum(tbrs) / len(tbrs), 1) if tbrs else None,
                "avg_cv_pct": round(sum(cvs) / len(cvs), 1) if cvs else None,
            }
        if metric == "hba1c":
            valid = [d for d in glucose_daily if d.get("estimatedHba1c") is not None]
            if not valid:
                return {"days_count": 0}
            max_d = max(valid, key=lambda d: d["estimatedHba1c"])
            min_d = min(valid, key=lambda d: d["estimatedHba1c"])
            return {
                "days_count": len(valid),
                "avg_hba1c_pct": round(sum(d["estimatedHba1c"] for d in valid) / len(valid), 2),
                "max_hba1c_pct": round(max_d["estimatedHba1c"], 2), "max_hba1c_date": max_d["glucoseDate"],
                "min_hba1c_pct": round(min_d["estimatedHba1c"], 2), "min_hba1c_date": min_d["glucoseDate"],
            }
        if metric == "fbs":
            valid = [d for d in glucose_daily if d.get("fbs") is not None]
            if not valid:
                return {"days_count": 0}
            max_d = max(valid, key=lambda d: d["fbs"])
            min_d = min(valid, key=lambda d: d["fbs"])
            return {
                "days_count": len(valid),
                "avg_fbs": round(sum(d["fbs"] for d in valid) / len(valid), 1),
                "max_fbs": round(max_d["fbs"], 1), "max_fbs_date": max_d["glucoseDate"],
                "min_fbs": round(min_d["fbs"], 1), "min_fbs_date": min_d["glucoseDate"],
            }
        if metric == "tir":
            best_day = max(glucose_daily, key=lambda d: d["tirPercent"])
            worst_day = min(glucose_daily, key=lambda d: d["tirPercent"])
            return {
                "days_count": len(glucose_daily),
                "avg_tir_pct": round(sum(d["tirPercent"] for d in glucose_daily) / len(glucose_daily), 1),
                "avg_tar_pct": round(sum(d["tarPercent"] for d in glucose_daily) / len(glucose_daily), 1),
                "avg_tbr_pct": round(sum(d["tbrPercent"] for d in glucose_daily) / len(glucose_daily), 1),
                "best_tir_pct": round(best_day["tirPercent"], 1), "best_tir_date": best_day["glucoseDate"],
                "worst_tir_pct": round(worst_day["tirPercent"], 1), "worst_tir_date": worst_day["glucoseDate"],
            }
        if metric == "sleep":
            sleep = data.get("sleep") or []
            if not sleep:
                return {"days_count": 0}
            totals = [(d, d.get("deepSleep", 0) + d.get("lightSleep", 0) + d.get("remSleep", 0)) for d in sleep]
            most = max(totals, key=lambda t: t[1])
            least = min(totals, key=lambda t: t[1])
            return {
                "days_count": len(sleep),
                "avg_light_hrs": round(sum(d.get("lightSleep", 0) for d in sleep) / len(sleep) / 60, 1),
                "avg_deep_hrs": round(sum(d.get("deepSleep", 0) for d in sleep) / len(sleep) / 60, 1),
                "avg_rem_hrs": round(sum(d.get("remSleep", 0) for d in sleep) / len(sleep) / 60, 1),
                "most_sleep_hrs": round(most[1] / 60, 1), "most_sleep_date": most[0]["readingDate"],
                "least_sleep_hrs": round(least[1] / 60, 1), "least_sleep_date": least[0]["readingDate"],
            }
        if metric == "activity":
            activity = data.get("activity") or []
            if not activity:
                return {"days_count": 0}
            steps = [d.get("totalSteps", 0) for d in activity]
            max_d = max(activity, key=lambda d: d.get("totalSteps", 0))
            min_d = min(activity, key=lambda d: d.get("totalSteps", 0))
            return {
                "days_count": len(activity),
                "avg_steps": round(sum(steps) / len(steps)),
                "max_steps": max_d.get("totalSteps", 0), "max_steps_date": max_d["readingDate"],
                "min_steps": min_d.get("totalSteps", 0), "min_steps_date": min_d["readingDate"],
                "avg_calories": round(sum(d.get("calories", 0) for d in activity) / len(activity)),
            }
        if metric == "bp":
            bp = data.get("bp") or []
            if not bp:
                return {"days_count": 0}
            max_sys = max(bp, key=lambda d: d.get("averageSystolic", 0))
            min_sys = min(bp, key=lambda d: d.get("averageSystolic", 0))
            return {
                "days_count": len(bp),
                "avg_systolic": round(sum(d.get("averageSystolic", 0) for d in bp) / len(bp), 1),
                "avg_diastolic": round(sum(d.get("averageDiastolic", 0) for d in bp) / len(bp), 1),
                "max_systolic": round(max_sys.get("averageSystolic", 0), 1), "max_systolic_date": max_sys["readingDate"],
                "min_systolic": round(min_sys.get("averageSystolic", 0), 1), "min_systolic_date": min_sys["readingDate"],
            }
        if metric == "heart_rate":
            hr = data.get("hr") or []
            if not hr:
                return {"days_count": 0}
            max_d = max(hr, key=lambda d: d.get("highHeartRate", 0))
            min_d = min(hr, key=lambda d: d.get("lowHeartRate", 0))
            return {
                "days_count": len(hr),
                "avg_hr_bpm": round(sum(d.get("averageHeartRate", 0) for d in hr) / len(hr), 1),
                "max_hr_bpm": max_d.get("highHeartRate", 0), "max_hr_date": max_d["readingDate"],
                "min_hr_bpm": min_d.get("lowHeartRate", 0), "min_hr_date": min_d["readingDate"],
            }
        if metric == "stress_hrv":
            stress = data.get("stress") or []
            hrv = data.get("hrv") or []
            result = {"days_count": max(len(stress), len(hrv))}
            if stress:
                result["avg_stress_pct"] = round(sum(d.get("average", 0) for d in stress) / len(stress), 1)
                max_s = max(stress, key=lambda d: d.get("average", 0))
                result["max_stress_pct"] = round(max_s.get("average", 0), 1)
                result["max_stress_date"] = max_s["readingDate"]
            else:
                result["avg_stress_pct"] = None
            if hrv:
                result["avg_hrv_ms"] = round(sum(d.get("average", 0) for d in hrv) / len(hrv), 1)
                min_h = min(hrv, key=lambda d: d.get("average", 0))
                result["min_hrv_ms"] = round(min_h.get("average", 0), 1)
                result["min_hrv_date"] = min_h["readingDate"]
            else:
                result["avg_hrv_ms"] = None
            return result
        return {}

    def _run_for_metric(self, metric: str, patient_id: Optional[int] = None,
                         patient_name: Optional[str] = None,
                         from_date: Optional[str] = None, to_date: Optional[str] = None,
                         period: Optional[str] = None) -> str:
        patient_id, resolution_error = self._resolve_patient_id(patient_id, patient_name)
        if resolution_error:
            object.__setattr__(self, 'last_chart_data', None)
            return resolution_error

        # Resolve whatever the caller gave (explicit dates, a month/year, or a
        # relative phrase) to a concrete window. NO hidden 7-day default: when
        # nothing is specified we analyse ALL available history. The API is a
        # /{from}/{to} path (it can't take "unbounded"), so all-history is a wide
        # floor window; the response still states the actual dates covered.
        start, end, period_label = resolve_range(from_date, to_date, period)
        if start:
            from_date, to_date = start, end
        else:
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now()
                         - timedelta(days=settings.TREND_ALL_HISTORY_LOOKBACK_DAYS)
                         ).strftime("%Y-%m-%d")

        user_context = getattr(self, 'user_context', None)
        auth_token = (user_context.get('auth_token') or user_context.get('token')) if user_context else None
        if not auth_token:
            return "Error: no authentication token available."

        try:
            data = self._fetch(patient_id, from_date, to_date, auth_token)
        except httpx.TimeoutException:
            return "The health data service timed out."
        except httpx.HTTPStatusError as e:
            return f"Could not retrieve health data (status {e.response.status_code})."
        except Exception as e:
            return f"Error retrieving health data: {str(e)}"

        glucose_daily = data.get("glucoseDailyAnalytics") or []
        status = "ok"
        notice = None
        # The old 180-day "most recent" fallback was removed here (rule 11: never
        # silently substitute a different period). An empty window is now reported
        # honestly by the metric branch below.

        program_summary = {}
        raw_program = data.get("patientProgramDetails")
        if raw_program:
            try:
                content = json.loads(raw_program).get("content", {})
                prog = content.get("patientProgram", {})
                details = content.get("patientDetails", {})
                program_summary = {
                    "doctor": prog.get("doctorName"),
                    "dha": prog.get("dhaName"),
                    "days_completed": prog.get("daysCompleted"),
                    "program_days": prog.get("programDays"),
                    "medications": prog.get("medications"),
                    "supplements": prog.get("supplements"),
                    "is_diabetic": details.get("isDiabetic"),
                }
            except Exception:
                pass

        metric_notice = None
        chart_data = None
        user_context = getattr(self, 'user_context', None)
        if len(glucose_daily) > 1:
            chart_data = self._build_chart(metric, glucose_daily, data, from_date, to_date)
            object.__setattr__(self, 'last_chart_data', chart_data)
            if user_context is not None and chart_data is not None:
                user_context['_last_trend_chart_data'] = chart_data
            if chart_data is None and metric != "glucose":
                metric_notice = (
                    f"No {metric} data was recorded for patient {patient_id} "
                    f"during {from_date} to {to_date}."
                )
        else:
            object.__setattr__(self, 'last_chart_data', None)

        base_payload = {
            "patient_id": patient_id,
            "date_range": f"{from_date} to {to_date}",
            "status": status,
            "notice": notice,
            "metric_notice": metric_notice,
        }

        RAW_FALLBACK_CAP = 20

        if metric == "tir":
            if chart_data is not None:
                base_payload["summary"] = self._summarize(metric, glucose_daily, data)
            else:
                base_payload["glucose_daily"] = glucose_daily[:RAW_FALLBACK_CAP]
                if len(glucose_daily) > RAW_FALLBACK_CAP:
                    base_payload["glucose_daily_note"] = f"Showing {RAW_FALLBACK_CAP} of {len(glucose_daily)} entries."
        elif metric == "hba1c":
            base_payload["summary"] = self._summarize(metric, glucose_daily, data)
        elif metric == "fbs":
            base_payload["summary"] = self._summarize(metric, glucose_daily, data)
        elif metric == "sleep":
            if chart_data is not None:
                base_payload["summary"] = self._summarize(metric, glucose_daily, data)
            else:
                raw_sleep = data.get("sleep") or []
                base_payload["sleep"] = raw_sleep[:RAW_FALLBACK_CAP]
                if len(raw_sleep) > RAW_FALLBACK_CAP:
                    base_payload["sleep_note"] = f"Showing {RAW_FALLBACK_CAP} of {len(raw_sleep)} entries."
        elif metric == "activity":
            if chart_data is not None:
                base_payload["summary"] = self._summarize(metric, glucose_daily, data)
            else:
                raw_activity = data.get("activity") or []
                base_payload["activity"] = raw_activity[:RAW_FALLBACK_CAP]
                if len(raw_activity) > RAW_FALLBACK_CAP:
                    base_payload["activity_note"] = f"Showing {RAW_FALLBACK_CAP} of {len(raw_activity)} entries."
        elif metric == "bp":
            if chart_data is not None:
                base_payload["summary"] = self._summarize(metric, glucose_daily, data)
            else:
                raw_bp = data.get("bp") or []
                base_payload["blood_pressure"] = raw_bp[:RAW_FALLBACK_CAP]
                if len(raw_bp) > RAW_FALLBACK_CAP:
                    base_payload["blood_pressure_note"] = f"Showing {RAW_FALLBACK_CAP} of {len(raw_bp)} entries."
        elif metric == "heart_rate":
            if chart_data is not None:
                base_payload["summary"] = self._summarize(metric, glucose_daily, data)
            else:
                raw_hr = data.get("hr") or []
                base_payload["heart_rate"] = raw_hr[:RAW_FALLBACK_CAP]
                if len(raw_hr) > RAW_FALLBACK_CAP:
                    base_payload["heart_rate_note"] = f"Showing {RAW_FALLBACK_CAP} of {len(raw_hr)} entries."
        elif metric == "stress_hrv":
            if chart_data is not None:
                base_payload["summary"] = self._summarize(metric, glucose_daily, data)
            else:
                base_payload["stress"] = (data.get("stress") or [])[:RAW_FALLBACK_CAP]
                base_payload["hrv"] = (data.get("hrv") or [])[:RAW_FALLBACK_CAP]
        elif metric == "glucose":
            summary = self._summarize("glucose", glucose_daily, data)
            if summary.get("days_count", 0) == 0:
                where = period_label if start else "this patient's available history"
                msg = f"No glucose data available for {where}."
                if user_context is not None:
                    user_context['_last_glucose_trend_text'] = msg
                return msg
            glucose_prose = _format_glucose_trend_prose(
                summary, from_date, to_date,
                is_fallback=(status != "ok"),
            )
            # Deliver verbatim, bypassing the LLM's formatting entirely (same
            # side-channel AGP/eHbA1c use). chat_routes returns this text directly
            # as the response, so the model cannot re-bullet it.
            if user_context is not None:
                user_context['_last_glucose_trend_text'] = glucose_prose
            return glucose_prose
        else:
            base_payload.update({
                "glucose_daily": glucose_daily[:RAW_FALLBACK_CAP],
                "blood_pressure": (data.get("bp") or [])[:RAW_FALLBACK_CAP],
                "heart_rate": (data.get("hr") or [])[:RAW_FALLBACK_CAP],
                "stress": (data.get("stress") or [])[:RAW_FALLBACK_CAP],
                "hrv": (data.get("hrv") or [])[:RAW_FALLBACK_CAP],
                "activity": (data.get("activity") or [])[:RAW_FALLBACK_CAP],
                "sleep": (data.get("sleep") or [])[:RAW_FALLBACK_CAP],
                "food_log_count": len(data.get("foodLogs") or []),
                "program_summary": program_summary
            })

        return json.dumps(base_payload)


_COMMON_TAIL = (
    " Requires EITHER patient_id (int) OR patient_name (str) — pass whichever "
    "the user gave you; do NOT guess a patient_id when only a name was given, "
    "this tool resolves the name itself. If multiple patients match the name, "
    "relay the returned list and ask the user which one — do not choose "
    "yourself. from_date/to_date (YYYY-MM-DD, or YYYY-MM / YYYY) are OPTIONAL, "
    "as is a 'period' phrase (e.g. 'last 30 days', 'July 2026', 'this month'); "
    "pass whichever the user gave. If the user names NO period at all, omit all "
    "three — the tool then analyses the patient's FULL available history; do NOT "
    "invent a default window. The result includes a 'status' field: 'ok', "
    "'ambiguous_name', 'not_found', or 'no_data' (no data for the requested "
    "period — the tool does NOT silently substitute another period). Always "
    "relay the 'notice' field verbatim when present. A chart is ALREADY shown "
    "automatically when data is found — do NOT write out a table or list of "
    "daily values yourself; give only a brief 1-3 sentence summary. The summary "
    "includes WHICH DATE had the highest/lowest/best/worst value — if the user "
    "asks a follow-up like 'which day was it', answer directly from the dated "
    "fields already in this result, do not call the tool again."
)


class GlucoseTrendTool(HealthProgressBase):
    name: str = "get_glucose_trend"
    description: str = (
        "Get a patient's glucose trend over a date range: mean glucose, "
        "TIR/TAR/TBR %, HbA1c estimate, FBS, plus BP, medications, and "
        "assigned doctor/DHA. Use for: 'glucose trend', 'sugar levels this "
        "week', 'BP for patient X', 'what medications', 'who is their "
        "doctor'." + _COMMON_TAIL
    )

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
              from_date: Optional[str] = None, to_date: Optional[str] = None,
              period: Optional[str] = None) -> str:
        return self._run_for_metric("glucose", patient_id, patient_name, from_date, to_date, period)


class TIRTrendTool(HealthProgressBase):
    name: str = "get_tir_trend"
    description: str = (
        "Get a patient's Time in Range (TIR) chart over a date range — "
        "percentage of time above/in-range/below target glucose. Use for: "
        "'time in range', 'TIR', 'TIR trend for patient X'." + _COMMON_TAIL
    )

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
              from_date: Optional[str] = None, to_date: Optional[str] = None) -> str:
        return self._run_for_metric("tir", patient_id, patient_name, from_date, to_date)


class SleepTrendTool(HealthProgressBase):
    name: str = "get_sleep_trend"
    description: str = (
        "Get a patient's sleep quality chart over a date range — deep, "
        "light, and REM sleep hours per day. Use for: 'sleep quality', "
        "'sleep trend', 'sleep pattern', 'sleep this week/for a range of "
        "dates'. For ONE single date only, use get_specific_medical_value "
        "instead." + _COMMON_TAIL
    )

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
              from_date: Optional[str] = None, to_date: Optional[str] = None) -> str:
        return self._run_for_metric("sleep", patient_id, patient_name, from_date, to_date)


class ActivityTrendTool(HealthProgressBase):
    name: str = "get_activity_trend"
    description: str = (
        "Get a patient's activity chart over a date range — daily steps and "
        "calories burned. Use for: 'activity', 'steps', 'activity summary/"
        "trend for patient X'." + _COMMON_TAIL
    )

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
              from_date: Optional[str] = None, to_date: Optional[str] = None) -> str:
        return self._run_for_metric("activity", patient_id, patient_name, from_date, to_date)


class HeartRateTrendTool(HealthProgressBase):
    name: str = "get_heart_rate_trend"
    description: str = (
        "Get a patient's heart rate chart over a date range — average, low, "
        "high, and resting heart rate per day. Use for: 'heart rate trend', "
        "'heart rate this week', 'heart rate for patient X'." + _COMMON_TAIL
    )

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
              from_date: Optional[str] = None, to_date: Optional[str] = None) -> str:
        return self._run_for_metric("heart_rate", patient_id, patient_name, from_date, to_date)


class StressHRVTrendTool(HealthProgressBase):
    name: str = "get_stress_hrv_trend"
    description: str = (
        "Get a patient's stress and HRV (Heart Rate Variability) chart over "
        "a date range. Use for: 'stress trend', 'HRV', 'stress and HRV for "
        "patient X'." + _COMMON_TAIL
    )

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
              from_date: Optional[str] = None, to_date: Optional[str] = None) -> str:
        return self._run_for_metric("stress_hrv", patient_id, patient_name, from_date, to_date)


class HbA1cTrendTool(HealthProgressBase):
    name: str = "get_hba1c_trend"
    description: str = (
        "Get a patient's estimated HbA1c trend chart over a date range. "
        "Use for: 'eHbA1c trend', 'HbA1c chart', 'HbA1c for patient X'." + _COMMON_TAIL
    )

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
              from_date: Optional[str] = None, to_date: Optional[str] = None) -> str:
        return self._run_for_metric("hba1c", patient_id, patient_name, from_date, to_date)


class FBSTrendTool(HealthProgressBase):
    name: str = "get_fbs_trend"
    description: str = (
        "Get a patient's Fasting Blood Sugar (FBS) trend chart over a date "
        "range. Use for: 'fasting blood sugar', 'FBS trend', 'FBS for "
        "patient X'." + _COMMON_TAIL
    )

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
              from_date: Optional[str] = None, to_date: Optional[str] = None) -> str:
        return self._run_for_metric("fbs", patient_id, patient_name, from_date, to_date)


class BPTrendTool(HealthProgressBase):
    name: str = "get_bp_trend"
    description: str = (
        "Get a patient's blood pressure chart over a date range — systolic "
        "and diastolic per day. Use for: 'blood pressure trend', 'BP chart', "
        "'BP this week', 'BP for patient X'." + _COMMON_TAIL
    )

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
              from_date: Optional[str] = None, to_date: Optional[str] = None) -> str:
        return self._run_for_metric("bp", patient_id, patient_name, from_date, to_date)