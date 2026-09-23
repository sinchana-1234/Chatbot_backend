"""
Response building for the medical chat routes.

Single source of truth for turning one agent turn into an API response, so the
/query and /voice routes never drift apart. Pure (no FastAPI/DB imports) and unit
tested.

How a turn's response text is decided
-------------------------------------
Deterministic tools (meal/activity/sleep/stress/lifestyle/glucose-trend/pattern)
each build their OWN finished, already-formatted answer and stash it on
``user_context`` under a key ending in ``_text`` (e.g. ``_last_meal_impact_text``).
We collect every such key generically:

  * exactly one deterministic tool produced text  -> return it verbatim
  * two or more did (a compound question)         -> return the LLM's own merged
                                                     message, so every factor shows
  * none did (a plain LLM answer)                 -> return the LLM's message

Because collection is by naming convention, adding a new deterministic tool needs
NO change in this file or the routes -- the tool just writes ``_last_<x>_text``.

Charts ride separate ``_data`` keys and are always passed through. Every key is
popped, so nothing leaks into a later, unrelated turn.
"""

from typing import Any, Dict, Optional, Tuple

_TREND_CHART_KEY = "_last_trend_chart_data"
_AGP_CHART_KEY = "_last_agp_chart_data"
_EHBA1C_CHART_KEY = "_last_ehba1c_tir_data"
_SUGGESTIONS_KEY = "_last_suggestions"

def _fallback_message(result: Any) -> str:
    """The LLM's own answer for this turn."""
    if isinstance(result, dict):
        return result.get("message", "") or ""
    return str(result) if result is not None else ""


def resolve_agent_output(
    user_context: Optional[Dict[str, Any]],
    result: Any,
) -> Tuple[str, Optional[dict], Optional[dict], Optional[dict], Optional[list]]:
    """
    Turn one agent turn into (response_text, chart_data, agp_chart_data,
    ehba1c_tir_data). Pops all consumed keys off ``user_context``.
    """
    fallback = _fallback_message(result)
    result_chart = result.get("chart_data") if isinstance(result, dict) else None

    if not user_context:
        return fallback, result_chart, None, None, None

    agp_chart_data = user_context.pop(_AGP_CHART_KEY, None)
    ehba1c_tir_data = user_context.pop(_EHBA1C_CHART_KEY, None)
    trend_chart_data = user_context.pop(_TREND_CHART_KEY, None)
    suggestions = user_context.pop(_SUGGESTIONS_KEY, None)
    chart_data = trend_chart_data or result_chart

    verbatim = []
    for key in [k for k in list(user_context.keys()) if k.endswith("_text")]:
        value = user_context.pop(key, None)
        if value:
            verbatim.append(value)

    response_text = verbatim[0] if len(verbatim) == 1 else fallback

    return response_text, chart_data, agp_chart_data, ehba1c_tir_data, suggestions