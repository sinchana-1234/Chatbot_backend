from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date


# ─────────────────────────────────────────────
# REQUEST SCHEMAS
# ─────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """
    Request to predict glucose for a patient.
    """

    patient_id: int = Field(..., description="Patient identifier")
    target_date: date = Field(..., description="Target date for prediction (YYYY-MM-DD)")
    target_hour: Optional[int] = Field(
        None,
        ge=0,
        le=23,
        description="Specific hour (0-23). If omitted, all 24 hours are returned."
    )


# ─────────────────────────────────────────────
# RESPONSE SCHEMAS
# ─────────────────────────────────────────────

class HourlyPrediction(BaseModel):
    hour: int
    predicted_mean: float
    predicted_max: float
    predicted_min: float


class InputLogEntry(BaseModel):
    """
    One minute-level CGM reading used as input.
    """

    timestamp: datetime
    glucose: float
    week_number: str
    hour: int
    minute: int
    dayofweek: int
    day_name: str


class WeekSummary(BaseModel):

    week_number: str
    week_start: date
    week_end: date

    total_readings: int

    overall_mean: float
    overall_max: float
    overall_min: float
    overall_std: float

    hypo_events: int
    hyper_events: int

    time_in_range_pct: float


class HourlyLogSummary(BaseModel):
    """
    Hourly statistics used for debugging / transparency.
    """

    hour: int
    week_number: str

    mean_glucose: float
    max_glucose: float
    min_glucose: float

    reading_count: int


class PredictionResponse(BaseModel):
    """
    Full response when prediction is possible.

    Pydantic V2 reserves the "model_" prefix for its own internals.
    protected_namespaces=() tells Pydantic that our "model_info" field
    is intentional — suppresses the UserWarning on startup.
    """

    model_config = {"protected_namespaces": ()}

    patient_id: str

    target_date: date
    target_day_name: str
    days_ahead: int

    predictions: List[HourlyPrediction]

    # Conceptual lag dates the model used (target - 14d, target - 7d).
    # Replaces input_weeks_used which was tied to ISO-week thinking.
    input_lag_dates: List[str]

    # Time in Range: % of the 24 predicted hourly means that fall within
    # the clinical safe range. Standard clinical metric.
    time_in_range_pct: float
    tir_range_mg_dl: List[int]  # [lower, upper] in mg/dL

    model_info: Optional[dict] = None


class SummaryOnlyResponse(BaseModel):
    """
    Returned when prediction cannot be made.
    """

    patient_id: str

    message: str

    available_weeks: int

    week_summaries: List[WeekSummary]

    raw_readings_sample: List[InputLogEntry] = []


class PatientWeeksResponse(BaseModel):

    patient_id: str

    available_weeks: int

    week_summaries: List[WeekSummary]

    has_enough_data_for_prediction: bool