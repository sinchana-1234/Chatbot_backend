#!/usr/bin/env python3
"""
Clinical Summary API
Fetches ALL available health data from healthProgress endpoint,
computes complete stats, and sends everything to GPT-4o-mini
to generate clinical status and observations.
"""

import os
import logging
import json
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from openai import OpenAI
import httpx

from auth.auth import get_current_user, UserContext, get_authorized_patient_id
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Clinical Summary"])

HEALTH_PROGRESS_BASE = settings.HEALTH_PROGRESS_BASE_URL


# ─── Helpers ─────────────────────────────────────────────────

def _safe_avg(values: list) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 2) if clean else None

def _safe_min(values: list) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(min(clean), 2) if clean else None

def _safe_max(values: list) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(max(clean), 2) if clean else None

def _split_halves(data: list, key: str):
    """Split list into first and second half, return avg of key for each."""
    if len(data) < 4:
        return None, None
    mid = len(data) // 2
    first  = [d[key] for d in data[:mid]  if d.get(key) is not None]
    second = [d[key] for d in data[mid:]  if d.get(key) is not None]
    first_avg  = round(sum(first)  / len(first),  2) if first  else None
    second_avg = round(sum(second) / len(second), 2) if second else None
    return first_avg, second_avg


def _compute_trend(glucose_data: list) -> dict:
    """
    Composite trend: eHbA1c (30%) + TIR (35%) + Mean glucose (35%).
    Negative composite = improving, Positive = worsening.
    """
    if len(glucose_data) < 4:
        return {
            "direction": "stable",
            "hba1c_first": None, "hba1c_second": None,
            "tir_first":   None, "tir_second":   None,
            "mean_first":  None, "mean_second":  None,
            "composite_score": None,
        }

    hba1c_first,  hba1c_second  = _split_halves(glucose_data, "estimatedHba1c")
    tir_first,    tir_second    = _split_halves(glucose_data, "tirPercent")
    mean_first,   mean_second   = _split_halves(glucose_data, "meanGlucose")

    hba1c_change = (hba1c_second - hba1c_first) if (hba1c_first  and hba1c_second)  else 0
    tir_change   = -(tir_second  - tir_first)   if (tir_first    and tir_second)    else 0
    mean_change  = (mean_second  - mean_first)  if (mean_first   and mean_second)   else 0

    hba1c_norm = hba1c_change / 2.0  if hba1c_change != 0 else 0
    tir_norm   = tir_change   / 30.0 if tir_change   != 0 else 0
    mean_norm  = mean_change  / 50.0 if mean_change  != 0 else 0

    composite = round(
        hba1c_norm * 0.30 +
        tir_norm   * 0.35 +
        mean_norm  * 0.35,
        4
    )

    if composite < -0.05:
        direction = "improving"
    elif composite > 0.05:
        direction = "worsening"
    else:
        direction = "stable"

    return {
        "direction":       direction,
        "hba1c_first":     hba1c_first,
        "hba1c_second":    hba1c_second,
        "tir_first":       tir_first,
        "tir_second":      tir_second,
        "mean_first":      mean_first,
        "mean_second":     mean_second,
        "composite_score": composite,
    }


# ─── Stats computation — ALL fields from API ─────────────────

def _compute_stats(
    glucose_data, bp_data, hr_data,
    stress_data, hrv_data, activity_data, sleep_data,
    from_date, to_date
) -> dict:
    """
    Extract and compute statistics from ALL available healthProgress fields.
    Every field from the API is included so the LLM has complete context.
    """
    trend = _compute_trend(glucose_data)

    # ── GLUCOSE — all fields ──────────────────────────────────
    avg_tir          = _safe_avg([d["tirPercent"]          for d in glucose_data])
    avg_tar          = _safe_avg([d["tarPercent"]          for d in glucose_data])
    avg_tbr          = _safe_avg([d["tbrPercent"]          for d in glucose_data])
    avg_glucose      = _safe_avg([d["meanGlucose"]         for d in glucose_data])
    avg_min_glucose  = _safe_avg([d["minGlucose"]          for d in glucose_data])
    avg_max_glucose  = _safe_avg([d["maxGlucose"]          for d in glucose_data])
    avg_std_dev      = _safe_avg([d["stdDeviation"]        for d in glucose_data])
    avg_cv           = _safe_avg([d["glycemicVariabilityCv"] for d in glucose_data])
    avg_hba1c        = _safe_avg([d["estimatedHba1c"]      for d in glucose_data if d.get("estimatedHba1c")])
    latest_hba1c     = max(glucose_data, key=lambda d: d["glucoseDate"])["estimatedHba1c"] if glucose_data else None
    # FBS — fasting blood sugar (may be null for some days)
    fbs_values       = [d["fbs"] for d in glucose_data if d.get("fbs") is not None]
    avg_fbs          = _safe_avg(fbs_values)
    min_fbs          = _safe_min(fbs_values)
    max_fbs          = _safe_max(fbs_values)
    fbs_days         = len(fbs_values)
    hyper_days       = sum(1 for d in glucose_data if d.get("tarPercent", 0) > 0)
    hypo_days        = sum(1 for d in glucose_data if d.get("tbrPercent", 0) > 0)
    total_days       = len(glucose_data)

    # ── BLOOD PRESSURE ────────────────────────────────────────
    avg_systolic     = _safe_avg([d["averageSystolic"]  for d in bp_data])
    avg_diastolic    = _safe_avg([d["averageDiastolic"] for d in bp_data])
    min_systolic     = _safe_min([d["averageSystolic"]  for d in bp_data])
    max_systolic     = _safe_max([d["averageSystolic"]  for d in bp_data])

    # ── HEART RATE ────────────────────────────────────────────
    avg_hr           = _safe_avg([d["averageHeartRate"] for d in hr_data])
    min_hr           = _safe_min([d["averageHeartRate"] for d in hr_data])
    max_hr           = _safe_max([d["averageHeartRate"] for d in hr_data])

    # ── STRESS ────────────────────────────────────────────────
    avg_stress       = _safe_avg([d["average"] for d in stress_data])
    max_stress       = _safe_max([d["average"] for d in stress_data])

    # ── HRV ───────────────────────────────────────────────────
    avg_hrv          = _safe_avg([d["average"] for d in hrv_data])
    min_hrv          = _safe_min([d["average"] for d in hrv_data])

    # ── ACTIVITY ──────────────────────────────────────────────
    avg_steps        = _safe_avg([d["totalSteps"] for d in activity_data])
    avg_calories     = _safe_avg([d["calories"]   for d in activity_data])
    max_steps        = _safe_max([d["totalSteps"] for d in activity_data])
    min_steps        = _safe_min([d["totalSteps"] for d in activity_data])

    # ── SLEEP — all three stages ───────────────────────────────
    avg_light_sleep  = _safe_avg([d.get("lightSleep", 0) for d in sleep_data])
    avg_deep_sleep   = _safe_avg([d.get("deepSleep",  0) for d in sleep_data])
    avg_rem_sleep    = _safe_avg([d.get("remSleep",   0) for d in sleep_data])
    avg_sleep_mins   = _safe_avg([
        d.get("lightSleep", 0) + d.get("deepSleep", 0) + d.get("remSleep", 0)
        for d in sleep_data
    ])

    return {
        "period": {"from": from_date, "to": to_date, "total_days": total_days},

        "glucose": {
            # Averages
            "avg_mean_glucose_mg_dl":     avg_glucose,
            "avg_min_glucose_mg_dl":      avg_min_glucose,
            "avg_max_glucose_mg_dl":      avg_max_glucose,
            # Time in ranges
            "avg_tir_pct":                avg_tir,
            "tir_first_half_pct":         trend["tir_first"],
            "tir_second_half_pct":        trend["tir_second"],
            "avg_tar_pct":                avg_tar,
            "avg_tbr_pct":                avg_tbr,
            # Variability
            "avg_std_deviation":          avg_std_dev,
            "avg_cv_pct":                 avg_cv,
            # eHbA1c
            "avg_estimated_hba1c_pct":    avg_hba1c,
            "latest_estimated_hba1c_pct": latest_hba1c,
            "hba1c_first_half_avg":       trend["hba1c_first"],
            "hba1c_second_half_avg":      trend["hba1c_second"],
            # Mean glucose trend
            "mean_first_half_avg":        trend["mean_first"],
            "mean_second_half_avg":       trend["mean_second"],
            # Composite trend
            "composite_trend_direction":  trend["direction"],
            "composite_trend_score":      trend["composite_score"],
            # Events
            "hyper_days":                 hyper_days,
            "hypo_days":                  hypo_days,
            # FBS
            "avg_fbs_mg_dl":              avg_fbs,
            "min_fbs_mg_dl":              min_fbs,
            "max_fbs_mg_dl":              max_fbs,
            "fbs_days_available":         fbs_days,
        },

        "blood_pressure": {
            "avg_systolic_mmhg":  avg_systolic,
            "avg_diastolic_mmhg": avg_diastolic,
            "min_systolic_mmhg":  min_systolic,
            "max_systolic_mmhg":  max_systolic,
        },

        "heart_rate": {
            "avg_bpm": avg_hr,
            "min_bpm": min_hr,
            "max_bpm": max_hr,
        },

        "stress": {
            "avg_score_out_of_100": avg_stress,
            "max_score_out_of_100": max_stress,
        },

        "hrv": {
            "avg_ms": avg_hrv,
            "min_ms": min_hrv,
        },

        "activity": {
            "avg_daily_steps":    avg_steps,
            "max_daily_steps":    max_steps,
            "min_daily_steps":    min_steps,
            "avg_daily_calories": avg_calories,
        },

        "sleep": {
            "avg_total_minutes":  avg_sleep_mins,
            "avg_total_hours":    round(avg_sleep_mins / 60, 1) if avg_sleep_mins else None,
            "avg_light_sleep_min": avg_light_sleep,
            "avg_deep_sleep_min":  avg_deep_sleep,
            "avg_rem_sleep_min":   avg_rem_sleep,
        },
    }


# ─── Clinical Status ──────────────────────────────────────────

def _compute_clinical_status(glucose_data: list) -> dict:
    """Phase determination using TIR and eHbA1c. 'since' removed."""
    if not glucose_data:
        return {
            "phase": None, "label": "Insufficient data",
            "avg_tir_pct": None, "tir_first": None, "tir_second": None,
            "avg_hba1c": None, "trend": _compute_trend([]),
        }

    avg_tir   = _safe_avg([d["tirPercent"] for d in glucose_data])
    avg_hba1c = _safe_avg([d["estimatedHba1c"] for d in glucose_data if d.get("estimatedHba1c")])
    trend     = _compute_trend(glucose_data)

    if avg_tir is not None and avg_tir >= 90 and (avg_hba1c is None or avg_hba1c < 6.5):
        phase = 4
    elif avg_tir is not None and avg_tir >= 70:
        phase = 3
    elif (avg_tir is not None and avg_tir >= 50) or (avg_hba1c and avg_hba1c <= 8.5 and trend["direction"] == "improving"):
        phase = 2
    else:
        phase = 1

    return {
        "phase":      phase,
        "avg_tir_pct": avg_tir,
        "tir_first":   trend["tir_first"],
        "tir_second":  trend["tir_second"],
        "avg_hba1c":   avg_hba1c,
        "trend":       trend,
    }


# ─── LLM clinical analysis ───────────────────────────────────

def _generate_clinical_analysis(stats: dict, patient_info: dict) -> dict:
    """Send ALL computed stats to GPT-4o-mini for clinical analysis."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping LLM analysis")
        return {"clinicalStatus": None, "observations": []}

    client = OpenAI(api_key=api_key, timeout=60.0)

    # Patient context
    patient_ctx = ""
    if patient_info:
        details = patient_info.get("patientDetails", {})
        program = patient_info.get("patientProgram", {})
        patient_ctx = f"""
Patient: {details.get('name', 'Unknown')}
Age: {details.get('age')} | Sex: {details.get('sex')} | Weight: {details.get('weight')} kg | Height: {details.get('height')} cm
Is Diabetic: {details.get('isDiabetic')} | Diabetic Duration: {details.get('diabeticDuration')}
On Insulin: {details.get('isOnInsulin')} | On Medication: {details.get('isOnMedication')}
Lab HbA1c: {details.get('hba1cValue')}% (dated {details.get('hba1cDate')})
Doctor: {program.get('doctorName')} | Health Coach: {program.get('dhaName')}
Medications: {program.get('medications')}
Supplements: {program.get('supplements')}
Program: {program.get('daysCompleted')} of {program.get('programDays')} days completed
"""

    system_prompt = """You are an expert clinical AI assistant specializing in diabetes and metabolic health.
You analyze patient health data and generate structured clinical assessments.

You must respond ONLY with a valid JSON object — no markdown, no backticks, no preamble, no explanation.

The JSON must follow this exact structure:
{
  "clinicalStatus": {
    "phase": <integer 1-4>,
    "avg_tir_pct": <float or null>,
    "tir_history": <float or null>,
    "tir_current": <float or null>,
    "avg_ehba1c": <float or null>,
    "trend": <"improving" | "worsening" | "stable">
  },
  "observations": [
    {
      "severity": <"red" | "amber" | "green" | "blue">,
      "message": <string>
    }
  ]
}

Phase definitions:
- Phase 4 (Excellent Control): TIR >= 90% AND eHbA1c < 6.5%
- Phase 3 (Stable): TIR >= 70%
- Phase 2 (Improvement): TIR >= 50% OR eHbA1c <= 8.5% with improving trend
- Phase 1 (Critical): TIR < 50% with no improving trend

Severity:
- red: urgent concern requiring immediate attention
- amber: moderate concern, monitor and consider intervention
- green: positive finding
- blue: exceptional finding, consider clinical review

TERMINOLOGY RULES (strictly follow):
- Always write eHbA1c (NOT HbA1c) for CGM-estimated values
- Only use HbA1c when referring to a lab blood test result
- Correct: eHbA1c improved from 10.09% to 7.67%
- Wrong: HbA1c improved from 10.09% to 7.67%

OBSERVATION RULES:
- Generate ONE observation per metric area
- Cover ALL of: glucose overview, eHbA1c trend, hyperglycemia, hypoglycemia,
  fasting blood sugar (if available), glycemic variability, blood pressure,
  heart rate, stress, HRV, activity, sleep stages
- Include actual numbers in every observation
- Be concise and clinically actionable"""

    g = stats["glucose"]
    s = stats["sleep"]
    a = stats["activity"]

    user_prompt = f"""Analyze the following patient health data and generate a complete clinical assessment.

{patient_ctx}

=== PERIOD: {stats['period']['from']} to {stats['period']['to']} ({stats['period']['total_days']} days) ===

GLUCOSE:
- Average mean glucose: {g['avg_mean_glucose_mg_dl']} mg/dL
- Average min glucose:  {g['avg_min_glucose_mg_dl']} mg/dL
- Average max glucose:  {g['avg_max_glucose_mg_dl']} mg/dL
- Time in Range (70-180): {g['avg_tir_pct']}%
  History (first half): {g['tir_first_half_pct']}% | Current (second half): {g['tir_second_half_pct']}%
- Time Above Range (>180): {g['avg_tar_pct']}%
- Time Below Range (<70):  {g['avg_tbr_pct']}%
- Glycemic variability (CV): {g['avg_cv_pct']}%  [target <36%]
- Std deviation: {g['avg_std_deviation']} mg/dL

eHbA1c (CGM estimated):
- Average: {g['avg_estimated_hba1c_pct']}%
- Latest:  {g['latest_estimated_hba1c_pct']}%
- History (first half avg): {g['hba1c_first_half_avg']}% | Current (second half avg): {g['hba1c_second_half_avg']}%
- Trend direction: {g['composite_trend_direction']}

Mean glucose trend:
- First half avg: {g['mean_first_half_avg']} mg/dL | Second half avg: {g['mean_second_half_avg']} mg/dL

Glycemic events:
- Days with hyperglycemia (TAR>0): {g['hyper_days']} of {stats['period']['total_days']}
- Days with hypoglycemia (TBR>0):  {g['hypo_days']} of {stats['period']['total_days']}

Fasting Blood Sugar (FBS):
- Available for {g['fbs_days_available']} of {stats['period']['total_days']} days
- Average FBS: {g['avg_fbs_mg_dl']} mg/dL  [normal <100, pre-diabetic 100-125, diabetic >=126]
- Min FBS: {g['min_fbs_mg_dl']} mg/dL | Max FBS: {g['max_fbs_mg_dl']} mg/dL

BLOOD PRESSURE:
- Average: {stats['blood_pressure']['avg_systolic_mmhg']}/{stats['blood_pressure']['avg_diastolic_mmhg']} mmHg
- Range: {stats['blood_pressure']['min_systolic_mmhg']}-{stats['blood_pressure']['max_systolic_mmhg']} mmHg systolic

HEART RATE:
- Average: {stats['heart_rate']['avg_bpm']} bpm
- Range: {stats['heart_rate']['min_bpm']}-{stats['heart_rate']['max_bpm']} bpm

STRESS:
- Average: {stats['stress']['avg_score_out_of_100']}/100
- Peak: {stats['stress']['max_score_out_of_100']}/100

HRV:
- Average: {stats['hrv']['avg_ms']} ms  [healthy >40ms]
- Lowest recorded: {stats['hrv']['min_ms']} ms

ACTIVITY:
- Average daily steps: {a['avg_daily_steps']}  [goal 8000+]
- Range: {a['min_daily_steps']}-{a['max_daily_steps']} steps/day
- Average calories burned: {a['avg_daily_calories']} kcal/day

SLEEP:
- Average total: {s['avg_total_hours']} hours ({s['avg_total_minutes']} min)  [target 7h+]
- Light sleep avg: {s['avg_light_sleep_min']} min
- Deep sleep avg:  {s['avg_deep_sleep_min']} min  [target 90+ min]
- REM sleep avg:   {s['avg_rem_sleep_min']} min  [target 90+ min]

Generate the complete clinical assessment JSON now."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=2500,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)

        if "clinicalStatus" not in result or "observations" not in result:
            raise ValueError("LLM response missing required keys")

        logger.info(f"LLM analysis: phase={result['clinicalStatus'].get('phase')}, observations={len(result['observations'])}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"LLM returned invalid JSON: {e}")
        return {"clinicalStatus": None, "observations": []}
    except Exception as e:
        logger.error(f"LLM clinical analysis failed: {e}")
        return {"clinicalStatus": None, "observations": []}


# ─── Endpoint ─────────────────────────────────────────────────

@router.get("/clinical-summary/{patient_id}")
async def get_clinical_summary(
    patient_id: int,
    from_date: str = Query(..., description="Start date YYYY-MM-DD"),
    to_date: str = Query(..., description="End date YYYY-MM-DD"),
    current_user: UserContext = Depends(get_current_user),
):
    # Validate patient access
    authorized_patient_id = get_authorized_patient_id(patient_id, current_user)
    if authorized_patient_id is None:
        authorized_patient_id = patient_id
    patient_id = authorized_patient_id

    # Validate dates
    try:
        datetime.strptime(from_date, "%Y-%m-%d")
        datetime.strptime(to_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Fetch from healthProgress API
    url = f"{HEALTH_PROGRESS_BASE}/{patient_id}/{from_date}/{to_date}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Health progress data source timed out.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Health progress data source returned {e.response.status_code}.")
    except Exception as e:
        logger.error(f"Failed to fetch healthProgress for patient {patient_id}: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch health progress data.")

    # Parse ALL arrays from response
    glucose_data  = data.get("glucoseDailyAnalytics", [])
    bp_data       = data.get("bp", [])
    hr_data       = data.get("hr", [])
    stress_data   = data.get("stress", [])
    hrv_data      = data.get("hrv", [])
    activity_data = data.get("activity", [])
    sleep_data    = data.get("sleep", [])

    # Parse patientProgramDetails
    program_details = {}
    raw_program = data.get("patientProgramDetails")
    if raw_program:
        try:
            parsed = json.loads(raw_program)
            program_details = parsed.get("content", {})
            # Remove sensitive fields that should never be in API response
            if "patientDetails" in program_details:
                program_details["patientDetails"].pop("notificationToken", None)
        except Exception:
            pass

    # Step 1 — Compute ALL stats
    stats = _compute_stats(
        glucose_data, bp_data, hr_data,
        stress_data, hrv_data, activity_data, sleep_data,
        from_date, to_date
    )

    # Step 2 — Rule-based clinical status (fallback)
    clinical_status = _compute_clinical_status(glucose_data)

    # Step 3 — LLM generates everything from complete stats
    import asyncio
    llm_result = await asyncio.to_thread(_generate_clinical_analysis, stats, program_details)

    # Merge LLM + rule-based clinical status
    llm_status = llm_result.get("clinicalStatus") or {}
    final_status = {
        "phase":        llm_status.get("phase") or clinical_status["phase"],
        "avg_tir_pct":  clinical_status["avg_tir_pct"],
        "tir_history":  clinical_status["tir_first"],
        "tir_current":  clinical_status["tir_second"],
        "avg_ehba1c":   clinical_status["avg_hba1c"],
        "trend":        clinical_status["trend"]["direction"],
    }

    g = stats["glucose"]
    s = stats["sleep"]

    return {
        "patientId": patient_id,
        "from":      from_date,
        "to":        to_date,

        "clinicalStatus": final_status,
        "observations":   llm_result.get("observations", []),

        "summaryStats": {
            "glucose": {
                "avgMeanGlucose":         g["avg_mean_glucose_mg_dl"],
                "avgMinGlucose":          g["avg_min_glucose_mg_dl"],
                "avgMaxGlucose":          g["avg_max_glucose_mg_dl"],
                "avgTirPct":              g["avg_tir_pct"],
                "tirHistoryPct":          g["tir_first_half_pct"],
                "tirCurrentPct":          g["tir_second_half_pct"],
                "avgTarPct":              g["avg_tar_pct"],
                "avgTbrPct":              g["avg_tbr_pct"],
                "avgCvPct":               g["avg_cv_pct"],
                "avgStdDeviation":        g["avg_std_deviation"],
                "avgEstimatedHba1c":      g["avg_estimated_hba1c_pct"],
                "latestEstimatedHba1c":   g["latest_estimated_hba1c_pct"],
                "avgFbs":                 g["avg_fbs_mg_dl"],
                "minFbs":                 g["min_fbs_mg_dl"],
                "maxFbs":                 g["max_fbs_mg_dl"],
                "fbsDaysAvailable":       g["fbs_days_available"],
            },
            "bloodPressure": {
                "avgSystolic":  stats["blood_pressure"]["avg_systolic_mmhg"],
                "avgDiastolic": stats["blood_pressure"]["avg_diastolic_mmhg"],
                "minSystolic":  stats["blood_pressure"]["min_systolic_mmhg"],
                "maxSystolic":  stats["blood_pressure"]["max_systolic_mmhg"],
            },
            "heartRate": {
                "avgHeartRate": stats["heart_rate"]["avg_bpm"],
                "minHeartRate": stats["heart_rate"]["min_bpm"],
                "maxHeartRate": stats["heart_rate"]["max_bpm"],
            },
            "stress": {
                "avgStress": stats["stress"]["avg_score_out_of_100"],
                "maxStress": stats["stress"]["max_score_out_of_100"],
            },
            "hrv": {
                "avgHrv": stats["hrv"]["avg_ms"],
                "minHrv": stats["hrv"]["min_ms"],
            },
            "activity": {
                "avgDailySteps":    stats["activity"]["avg_daily_steps"],
                "maxDailySteps":    stats["activity"]["max_daily_steps"],
                "minDailySteps":    stats["activity"]["min_daily_steps"],
                "avgDailyCalories": stats["activity"]["avg_daily_calories"],
            },
            "sleep": {
                "avgSleepMinutes":   s["avg_total_minutes"],
                "avgSleepHours":     s["avg_total_hours"],
                "avgLightSleepMin":  s["avg_light_sleep_min"],
                "avgDeepSleepMin":   s["avg_deep_sleep_min"],
                "avgRemSleepMin":    s["avg_rem_sleep_min"],
            },
        },

        "patientProgram": program_details,
    }