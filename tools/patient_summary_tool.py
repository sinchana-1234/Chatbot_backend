import logging
from typing import Optional
from datetime import datetime, timedelta, date as date_cls
from langchain.tools import BaseTool

from dal.postgres_queries import get_mobile_metrics
from config import settings

logger = logging.getLogger(__name__)


def fmt(value, unit=""):
    if value is None:
        return "Not recorded"
    if isinstance(value, str):
        return f"{value} {unit}".strip()
    try:
        return f"{round(float(value), 1)} {unit}".strip()
    except:
        return str(value)


def fmt_int(value, unit=""):
    """Format integer values like steps without decimals."""
    if value is None:
        return "Not recorded"
    try:
        return f"{int(value):,} {unit}".strip()
    except:
        return str(value)


def format_sleep(minutes, query_type):
    if minutes is None:
        return "Not recorded"
    try:
        minutes = int(minutes)
        hours = minutes // 60
        mins = minutes % 60
        base = f"{hours}h {mins}m"
        if query_type in ("overall", "weekly"):
            return f"~{base} / night average"
        return base   # daily or specific date — just show the raw duration
    except:
        return str(minutes)


class PatientSummaryTool(BaseTool):
    name: str = "get_patient_summary"
    description: str = "Get patient health summary (today, yesterday, overall)"

    def set_user_context(self, user_context):
        object.__setattr__(self, "user_context", user_context)

    def _resolve_date(self, input_date: Optional[str]):
        if not input_date:
            return None
        input_date = input_date.lower().strip()
        if input_date == "today":
            return date_cls.today()
        if input_date == "yesterday":
            return date_cls.today() - timedelta(days=1)
        try:
            return datetime.strptime(input_date, "%Y-%m-%d").date()
        except:
            return None

    def _get_period_label(self, query_type, target_date_obj):
        if query_type == "overall":
            return "Lifetime Health Dashboard"
        if query_type == "weekly":
            return "Weekly Health Dashboard"
        if query_type == "daily" or target_date_obj:
            if target_date_obj:
                formatted = target_date_obj.strftime("%B %d, %Y")
                if target_date_obj == date_cls.today():
                    return f"Today's Summary ({formatted})"
                if target_date_obj == date_cls.today() - timedelta(days=1):
                    return f"Yesterday's Summary ({formatted})"
                return f"Summary for {formatted}"
        return "Health Summary"

    def _get_overall_status(self, glucose, bp, spo2, stress):
        # If none of the key metrics have any data, don't show a health status at all
        if glucose is None and bp is None and spo2 is None and stress is None:
            return "⚪", "No Data Available", "No readings were recorded for this period."

        issues = 0
        if glucose and (glucose > settings.GLUCOSE_HIGH_MGDL or glucose < settings.GLUCOSE_LOW_MGDL):
            issues += 1
        if bp and "/" in str(bp):
            try:
                sys, dia = map(int, bp.split("/"))
                if sys > 130 or dia > 85:
                    issues += 1
            except:
                pass
        if spo2 and spo2 < 95:
            issues += 1
        if stress and stress > 60:
            issues += 1

        if issues == 0:
            return "🟢", "Excellent Performance", "Great job! You are consistently hitting your health targets."
        elif issues == 1:
            return "🟡", "Good with Minor Concerns", "Most of your readings are in range. Keep an eye on flagged metrics."
        else:
            return "🔴", "Needs Attention", "Some of your readings need attention. Please consult your healthcare provider."

    def _get_condition_text(self, glucose, bp, spo2, stress):
        # If no clinical metrics have data, say so explicitly
        if glucose is None and bp is None and spo2 is None and stress is None:
            return "No clinical readings available for this period."

        severe = []
        mild = []

        if glucose:
            _normal = f"{settings.GLUCOSE_LOW_MGDL}–{settings.GLUCOSE_HIGH_MGDL}"
            if glucose > settings.GLUCOSE_VERY_HIGH_MGDL:
                severe.append(f"Glucose is very high at {fmt(glucose, 'mg/dL')} (normal: {_normal}). Consult your doctor.")
            elif glucose > settings.GLUCOSE_HIGH_MGDL:
                severe.append(f"Glucose is high at {fmt(glucose, 'mg/dL')} (normal: {_normal}). Consult your doctor.")
            elif glucose < settings.GLUCOSE_LOW_MGDL:
                severe.append(f"Glucose is low at {fmt(glucose, 'mg/dL')} (normal: {_normal}). Have a snack and monitor.")

        if bp and "/" in str(bp):
            try:
                sys, dia = map(int, bp.split("/"))
                if sys > 140 or dia > 90:
                    severe.append(f"Blood pressure high at {sys}/{dia} mmHg. Please consult your doctor.")
                elif sys > 130 or dia > 85:
                    mild.append(f"Blood pressure slightly elevated at {sys}/{dia} mmHg. Reduce salt and stress.")
            except:
                pass

        if spo2 and spo2 < 95:
            severe.append(f"Oxygen level low at {fmt(spo2, '%')}. Seek medical attention if breathless.")

        if stress:
            if stress > 80:
                severe.append(f"Stress very high ({fmt(stress, '/100')}). Rest and consider speaking to someone.")
            elif stress > 60:
                mild.append(f"Stress is moderate ({fmt(stress, '/100')}). Try breathing exercises or a short walk.")

        if severe:
            return "\n".join([f"  • {i}" for i in severe])
        if mild:
            return "\n".join([f"  • {i}" for i in mild])
        return "All clinical parameters are within normal range."

    def _run(
        self,
        patient_id: Optional[int] = None,
        query_type: Optional[str] = None,
        date: Optional[str] = None,                  # agent prompt says 'date='
        summary_date_input: Optional[str] = None,    # LLM sometimes infers this from old signature
        summary_date: Optional[str] = None,
        input_date: Optional[str] = None,
        **kwargs
    ) -> str:
        try:
            user_context = getattr(self, "user_context", None)
            # Role-based access control:
            #  - Patient (role_id == 1): ALWAYS force their own id — a patient can
            #    only ever see their own summary.
            #  - Staff: keep the patient_id they asked for; if they named none,
            #    default to their own id (self-summary), like the other tools.
            # The old code forced EVERY logged-in user to their own id, so staff
            # could never open a patient's summary.
            if user_context and user_context.get("role_id") == 1:   # Patient
                patient_id = user_context.get("user_id")
            elif not patient_id and user_context:                   # Staff, none given
                patient_id = user_context.get("user_id")
            if not patient_id:
                return "Patient ID not found."

            # Accept date from any of the possible parameter names the LLM might use
            resolved_input = date or summary_date_input or summary_date or input_date
            target_date_obj = self._resolve_date(resolved_input)
            target_date = target_date_obj.strftime("%Y-%m-%d") if target_date_obj else None

            start_date = None
            if query_type == "overall":
                # All-time averages — no date filter
                target_date = None
            elif query_type == "weekly":
                # Last 7 days averages
                start_date = (date_cls.today() - timedelta(days=6)).strftime("%Y-%m-%d")
                target_date = None
            elif query_type == "daily":
                # Specific date — target_date already set above from 'date' param
                # If no date was passed alongside daily, default to today
                if not target_date:
                    target_date = date_cls.today().strftime("%Y-%m-%d")
                    target_date_obj = date_cls.today()

            data = get_mobile_metrics(patient_id, target_date, start_date)
            if not data:
                return "No health data available."

            glucose  = data.get("glucose")
            bp       = data.get("bp")
            temp     = data.get("temperature")
            spo2     = data.get("spo2")
            hr       = data.get("heart_rate")
            hrv      = data.get("hrv")
            stress   = data.get("stress")
            sleep    = data.get("sleep")
            steps    = data.get("steps")
            calories = data.get("calories_burned")
            active   = data.get("active_time")

            period_label = self._get_period_label(query_type, target_date_obj)
            status_emoji, status_text, status_desc = self._get_overall_status(glucose, bp, spo2, stress)
            clinical_text = self._get_condition_text(glucose, bp, spo2, stress)

            # Format active time in minutes label
            active_label = f"{int(active)} mins" if active is not None else "Not recorded"

            return f"""**{period_label}**

{status_emoji} Status: {status_text}

*{status_desc}*

---

📊 **Vital Signs at a Glance**

• 🩸 **Glucose:** `{fmt(glucose, "mg/dL")}`
• 🫀 **Blood Pressure:** `{fmt(bp, "mmHg")}`
• ❤️ **Heart Rate:** `{fmt(hr, "bpm")}`

---

🏃 **Activity & Movement**

• 🚶 **Steps:** `{fmt_int(steps)}`
• 🔥 **Calories Burned:** `{fmt(calories, "kcal")}`
• ⏱️ **Active Time:** `{active_label}`

---

🛌 **Recovery & Wellness**

• 💤 **Average Sleep:** `{format_sleep(sleep, query_type)}`
• 📈 **Heart Rate Variability (HRV):** `{fmt(hrv, "ms")}`
• 🧠 **Stress Level:** `{fmt(stress, "/100")}`

---

📋 **Clinical Insights & Analysis**

{clinical_text}"""

        except Exception as e:
            logger.error(f"Error generating patient summary: {e}")
            return "Error generating patient summary."