"""
FastAPI Router — CGM Glucose Prediction Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import date
from typing import Optional
import logging

from database import get_db
from schemas import (
    PredictionRequest,
    PredictionResponse,
    SummaryOnlyResponse,
    PatientWeeksResponse,
    HourlyPrediction,
    WeekSummary,
)

from prediction_service import prediction_service

from feature_engineering import (
    clean_cgm_data,
    build_hourly_dataframe,
    add_time_features,
    engineer_features,
    compute_week_summary,
)

from config import settings

router = APIRouter(prefix="/api/v1", tags=["Glucose Prediction"])
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# GET /patients/{patient_id}/weeks
# ─────────────────────────────────────────────
@router.get(
    "/patients/{patient_id}/weeks",
    response_model=PatientWeeksResponse,
)
def get_patient_weeks(patient_id: int, db: Session = Depends(get_db)):

    # Full history — UI needs all weeks the patient ever wore the sensor
    df_raw = prediction_service.load_patient_data(db, patient_id, limit_days=None)

    if df_raw.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for patient '{patient_id}'"
        )

    df_clean = clean_cgm_data(df_raw)
    df_hourly = build_hourly_dataframe(df_clean)
    df_time = add_time_features(df_hourly)

    available_weeks = sorted(df_time["week_number"].unique())

    week_summaries = [
        WeekSummary(**summary)
        for w in available_weeks
        if (summary := compute_week_summary(df_time, w))
    ]

    return PatientWeeksResponse(
        patient_id=str(patient_id),
        available_weeks=len(available_weeks),
        week_summaries=week_summaries,
        has_enough_data_for_prediction=len(available_weeks)
        >= settings.MIN_WEEKS_FOR_PREDICTION,
    )


# ─────────────────────────────────────────────
# POST /predict
# ─────────────────────────────────────────────
@router.post("/predict")
def predict_glucose(
    request: PredictionRequest,
    db: Session = Depends(get_db),
):

  
    df_raw = prediction_service.load_patient_data(db, request.patient_id, limit_days=30)

    if df_raw.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No CGM data found for patient '{request.patient_id}'"
        )

    try:

        result = prediction_service.predict(
            df_raw=df_raw,
            patient_id=str(request.patient_id),
            target_date=request.target_date,
            target_hour=request.target_hour,
        )

        # ─────────────────────────────
        # Historical week requested
        # ─────────────────────────────
        if result.get("type") == "historical_summary":

            summary = result.get("week_summary")

            return SummaryOnlyResponse(
                patient_id=result["patient_id"],
                message=f"Historical summary for week {result['requested_week']}",
                available_weeks=result.get("available_weeks", 1),
                week_summaries=[WeekSummary(**summary)] if summary else [],
                raw_readings_sample=[],
            )

        # ─────────────────────────────
        # Missing consecutive weeks
        # ─────────────────────────────
        if result.get("type") == "summary_only":

            return SummaryOnlyResponse(
                patient_id=str(result["patient_id"]),
                message="Prediction unavailable due to missing consecutive weeks.",
                available_weeks=result.get("available_weeks", 1),
                week_summaries=[
                    WeekSummary(**w)
                    for w in result.get("week_summaries", [])
                    if w
                ],
                raw_readings_sample=[],
            )

        # ─────────────────────────────
        # Valid prediction
        # ─────────────────────────────
        return PredictionResponse(
            patient_id=str(result["patient_id"]),
            target_date=request.target_date,
            target_day_name=result["target_day_name"],
            days_ahead=result["days_ahead"],
            predictions=[
                HourlyPrediction(**p)
                for p in result.get("predictions", [])
            ],
            input_lag_dates=result.get("input_lag_dates", []),
            time_in_range_pct=result.get("time_in_range_pct", 0.0),
            tir_range_mg_dl=result.get("tir_range_mg_dl", [70, 180]),
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction error")


# # ─────────────────────────────────────────────
# # GET /predict (browser testing)
# # ─────────────────────────────────────────────
# @router.get("/predict")
# def predict_glucose_get(
#     patient_id: int = Query(...),
#     target_date: date = Query(...),
#     target_hour: Optional[int] = Query(None, ge=0, le=23),
#     db: Session = Depends(get_db),
# ):
#     return predict_glucose(
#         PredictionRequest(
#             patient_id=patient_id,
#             target_date=target_date,
#             target_hour=target_hour,
#         ),
#         db=db,
#     )


# ─────────────────────────────────────────────
# GET /patients/{patient_id}/summary
# ─────────────────────────────────────────────
@router.get(
    "/patients/{patient_id}/summary",
    response_model=SummaryOnlyResponse,
)
def get_patient_summary(patient_id: int, db: Session = Depends(get_db)):

    # Full history — summary page shows all historical weeks
    df_raw = prediction_service.load_patient_data(db, patient_id, limit_days=None)

    if df_raw.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data for patient '{patient_id}'"
        )

    df_clean = clean_cgm_data(df_raw)
    df_hourly = build_hourly_dataframe(df_clean)
    df_time = add_time_features(df_hourly)

    available_weeks = sorted(df_time["week_number"].unique())

    week_summaries = [
        WeekSummary(**summary)
        for w in available_weeks
        if (summary := compute_week_summary(df_time, w))
    ]

    return SummaryOnlyResponse(
        patient_id=str(patient_id),
        message="Summary of all available CGM data",
        available_weeks=len(available_weeks),
        week_summaries=week_summaries,
        raw_readings_sample=[],
    )