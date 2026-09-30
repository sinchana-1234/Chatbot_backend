#!/usr/bin/env python3
"""Patient Concerns / Summary Tool.

Answers open-ended "what are the main concerns / key points to watch / notable
findings / what should I follow up on" questions with a DETERMINISTIC, glucose-first
summary — so the router no longer falls through to the sleep/stress/activity tools,
and follow-up questions never get LLM-invented recommendations.

THIN ORCHESTRATOR — it re-implements NO glucose maths. It calls the existing
SpecificMedicalValueTool (overview + pattern_high + pattern_low) and reuses
meal_glucose_tool._correlate() for the meal rise, then assembles their REAL
results into one short summary. The `framing` argument only changes the PRESENTATION
of that same data: 'patterns' (a watch-list) or 'follow_up' (factual review areas).
"""

import logging
from typing import Optional
from collections import defaultdict
from langchain.tools import BaseTool

from tools.specific_medical_value_tool import (
    SpecificMedicalValueTool,
    _pattern_band_of, _pattern_top_bands, _pattern_join,
)

logger = logging.getLogger(__name__)


def _bands_from_pattern(hourly_pattern):
    """Reuse the SAME band vocabulary the pattern tool uses. Returns a phrase like
    'overnight and early morning', or None when there are no episodes."""
    episodes = [e for e in (hourly_pattern or []) if e.get("distinct_days", 0) >= 1]
    if not episodes:
        return None
    # Frequency first, mirroring _format_pattern_prose's ordering.
    shown = sorted(episodes, key=lambda e: (-e["distinct_days"], e["hour_of_day"]))
    bands = _pattern_top_bands([_pattern_band_of(e["hour_of_day"]) for e in shown], 3)
    return _pattern_join(bands)


def _top_meal(meals):
    """Meal type with the largest AVERAGE rise. Mirrors _format_meal_impact's own
    grouping (avg per type, pick the max); the heavy lifting (_correlate SQL) is
    reused, this is only the pick."""
    measured = [m for m in (meals or []) if m.get("rise") is not None]
    if not measured:
        return None
    groups = defaultdict(list)
    for m in measured:
        groups[m["meal_type"]].append(m)
    stats = [{"type": mt, "avg": round(sum(i["rise"] for i in items) / len(items))}
             for mt, items in groups.items()]
    stats.sort(key=lambda s: s["avg"], reverse=True)
    return stats[0]


def _format_concerns(name, overview, low_bands, high_bands, meal, framing="patterns"):
    """Pure assembly. Two deterministic presentations of the SAME data:
    'patterns' — a watch-list; 'follow_up' — factual review areas (no recommendations)."""
    name = name or "This patient"
    lo = (overview or {}).get("lowest")
    hi = (overview or {}).get("highest")

    if framing == "follow_up":
        lines = [f"Potential follow-up areas for {name}:", ""]
        if lo is not None:
            where = f", particularly those during the {low_bands} hours" if low_bands else ""
            lines.append(f"- Low glucose: Review the low-glucose episodes{where}. "
                         f"Lowest recorded glucose: {int(round(lo))} mg/dL.")
        if hi is not None:
            band = f"{high_bands} " if high_bands else ""
            lines.append(f"- High glucose: Review the {band}high-glucose episodes. "
                         f"Highest recorded glucose: {int(round(hi))} mg/dL.")
        if meal:
            mt = meal["type"].lower()
            lines.append(f"- Meal response: Review the larger glucose rises observed after {mt}. "
                         f"Average {mt}-related rise: +{meal['avg']} mg/dL.")
        if len(lines) <= 2:
            return f"There isn't enough glucose data to identify follow-up areas for {name} yet."
        lines.append("- Lifestyle associations: Review the observed associations between "
                     "glucose and activity, sleep, and stress.")
        lines.append("")
        lines.append("These are areas identified from the available data and do not by "
                     "themselves establish the cause of the observed glucose patterns.")
        return "\n".join(lines)

    # default: 'patterns' watch-list
    lines = [f"Key points to watch for {name}:", ""]
    if lo is not None:
        tail = f", with low episodes mainly during the {low_bands} hours" if low_bands else ""
        lines.append(f"* Lowest glucose: {int(round(lo))} mg/dL{tail}.")
    if hi is not None:
        tail = f", with high episodes mainly during the {high_bands} hours" if high_bands else ""
        lines.append(f"* Highest glucose: {int(round(hi))} mg/dL{tail}.")
    if meal:
        lines.append(f"* Meal response: {meal['type']} showed the largest average glucose "
                     f"rise, about +{meal['avg']} mg/dL.")
    if len(lines) <= 2:
        return f"There isn't enough glucose data to flag concerns for {name} yet."
    return "\n".join(lines)


class PatientConcernsTool(BaseTool):
    name: str = "get_patient_concerns"
    description: str = (
        "Give a DETERMINISTIC, glucose-first summary of a patient's data: the glucose "
        "extremes (lowest/highest), when low and high episodes occur, and the meal with "
        "the largest average glucose rise. Requires patient_id (int) OR patient_name (str). "
        "Set 'framing' to control the presentation of the SAME data: "
        "framing='patterns' (DEFAULT) — a watch-list, for 'what are the main patterns', "
        "'main concerns', 'key concerns', 'notable findings', 'anything concerning', "
        "'important findings', 'what stands out'. "
        "framing='follow_up' — factual review areas (no recommendations), for 'what should "
        "I follow up on', 'what should I monitor', 'what needs follow-up', 'what should be "
        "followed up'. "
        "Do NOT use for lifestyle-only questions (sleep/stress/activity) or for trend charts."
    )

    def set_user_context(self, user_context):
        object.__setattr__(self, 'user_context', user_context)

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
             framing: str = "patterns") -> str:
        uc = getattr(self, 'user_context', None)
        try:
            spec = SpecificMedicalValueTool()
            if uc is not None:
                spec.set_user_context(uc)

            # 1) extremes — reuses the overview aggregate (real min/max), then read
            #    the structured copy it stashes.
            spec._run(patient_id=patient_id, patient_name=patient_name,
                      reading_type="glucose", analysis_type="overview")
            overview = uc.pop('_last_overview_data', None) if uc else None

            # 2) low + high episode timing — reuses the pattern bucketing.
            spec._run(patient_id=patient_id, patient_name=patient_name,
                      reading_type="glucose", analysis_type="pattern_low")
            low_data = uc.pop('_last_pattern_low_data', None) if uc else None
            spec._run(patient_id=patient_id, patient_name=patient_name,
                      reading_type="glucose", analysis_type="pattern_high")
            high_data = uc.pop('_last_pattern_high_data', None) if uc else None

            low_bands = _bands_from_pattern((low_data or {}).get("hourly_pattern"))
            high_bands = _bands_from_pattern((high_data or {}).get("hourly_pattern"))

            # resolved id comes back with the overview data; fall back to the arg.
            pid = (overview or {}).get("patient_id") or patient_id

            # 3) meal with the largest average rise — reuses _correlate (real SQL).
            meal = None
            if pid:
                try:
                    from tools.meal_glucose_tool import _correlate
                    from dal.cycle_window import cycle_window
                    cw = cycle_window(pid)
                    start, end = (cw if cw else (None, None))
                    meal = _top_meal(_correlate(pid, start, end))
                except Exception as e:
                    logger.warning(f"concerns: meal impact skipped: {e}")

            # resolve a display name (same lookup the pattern branch uses)
            name = None
            if pid:
                try:
                    from dal.database import DatabaseManager
                    with DatabaseManager() as dm:
                        u = dm.get_users(user_id=pid)
                        if u:
                            name = f"{u[0].first_name or ''} {u[0].last_name or ''}".strip() or None
                except Exception:
                    name = None

            # leave ONLY _last_concerns_text so response_builder delivers this
            # verbatim (a lingering *_text from a sub-call would trigger the
            # "2+ tools" concatenation path and dump raw sub-answers).
            if uc is not None:
                for k in ("_last_overview_text", "_last_pattern_text",
                          "_last_meal_impact_text", "_last_suggestions",
                          "_last_trend_chart_data", "_last_overview_data",
                          "_last_pattern_low_data", "_last_pattern_high_data"):
                    uc.pop(k, None)

            framing = framing if framing in ("patterns", "follow_up") else "patterns"
            summary = _format_concerns(name, overview, low_bands, high_bands, meal, framing)
            if uc is not None:
                uc["_last_concerns_text"] = summary
            return summary

        except Exception as e:
            logger.error(f"Error in PatientConcernsTool: {e}")
            return f"Error building the concerns summary: {str(e)}"

    async def _arun(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
                    framing: str = "patterns") -> str:
        return self._run(patient_id, patient_name, framing)