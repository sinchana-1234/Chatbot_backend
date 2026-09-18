from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("POSTGRES_DATABASE_URL") or os.getenv("DATABASE_URL")

# Apply SSL only for managed cloud Postgres (Azure, AWS RDS, GCP).
# Local/QA Postgres at internal IPs (e.g. 10.0.0.15) typically doesn't
# support SSL — applying these args there breaks the connection.
# This mirrors the logic in /database.py so all three Postgres engines
# behave the same way against cloud vs internal DBs.
_needs_ssl = any(host in (DATABASE_URL or "") for host in [
    ".database.azure.com",
    ".rds.amazonaws.com",
    ".gcp.cloud.sql",
])
_connect_args = {
    "sslmode": "require",
    "sslrootcert": "/certs/DigiCertGlobalRootG2.crt.pem",
} if _needs_ssl else {}

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    connect_args=_connect_args,
) if DATABASE_URL else None
SessionLocalPG = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None


def get_mobile_metrics(patient_id, target_date=None, start_date=None, end_date=None):
    """
    Fetch aggregated health metrics for a patient.

    Args:
        patient_id  : required
        target_date : 'YYYY-MM-DD' string  → fetch that single day
        start_date  : 'YYYY-MM-DD' string  → fetch averages from start_date
                      if end_date also passed → fetch from start_date to end_date (both inclusive)
                      if end_date not passed  → fetch from start_date up to today
                      (used for weekly: pass 7 days ago)
        end_date    : 'YYYY-MM-DD' string  → upper bound for custom date range (inclusive)
                      only used when start_date is also passed
    """
    if SessionLocalPG is None:
        return {}
    db = SessionLocalPG()

    try:
        results = {}

        # ─── Time-attribution policy ─────────────────────────────────────
        # We filter on the patient's LOCAL time, not UTC, so that "May 25"
        # means "May 25 in the patient's timezone" regardless of where the
        # server runs or where the patient lives. The schema stores both:
        #   - actual_time / local_event_time → patient's wall-clock time
        #   - date_time_utc  / event_time_utc → UTC equivalent
        # All rows have the local-time column populated (verified against
        # the live DB), so we can use it directly with no fallback.
        #
        # Glucose uses `local_event_time`; everything else uses `actual_time`.
        # Sleep uses `actual_time + INTERVAL '12 hours'` so a sleep session
        # that crosses midnight gets attributed to the wake-up date.
        # ─────────────────────────────────────────────────────────────────
        if target_date:
            # Single specific day
            date_filter_glucose  = "DATE(local_event_time) = :target_date"
            date_filter_other    = "DATE(actual_time) = :target_date"
            date_filter_activity = "DATE(actual_time) = :target_date"
            params = {"patient_id": patient_id, "target_date": target_date}
        elif start_date and end_date:
            # Custom date range: from start_date to end_date (both inclusive)
            date_filter_glucose  = "DATE(local_event_time) BETWEEN :start_date AND :end_date"
            date_filter_other    = "DATE(actual_time) BETWEEN :start_date AND :end_date"
            date_filter_activity = "DATE(actual_time) BETWEEN :start_date AND :end_date"
            params = {"patient_id": patient_id, "start_date": start_date, "end_date": end_date}
        elif start_date:
            # Date range from start_date to today (e.g. last 7 days for weekly)
            date_filter_glucose  = "DATE(local_event_time) >= :start_date"
            date_filter_other    = "DATE(actual_time) >= :start_date"
            date_filter_activity = "DATE(actual_time) >= :start_date"
            params = {"patient_id": patient_id, "start_date": start_date}
        else:
            # All-time overall
            date_filter_glucose  = "1=1"
            date_filter_other    = "1=1"
            date_filter_activity = "1=1"
            params = {"patient_id": patient_id}

        # Sleep queries do their own filtering — they need a +12h shift so a
        # sleep session that crosses midnight gets attributed to the wake-up
        # date. The shared date_filter_* variables above don't apply to sleep.

        # GLUCOSE
        results["glucose"] = db.execute(text(f"""
            SELECT AVG(glucose_value)
            FROM glucose_readings
            WHERE patient_id = :patient_id
            AND {date_filter_glucose}
        """), params).scalar()

        # BLOOD PRESSURE
        bp = db.execute(text(f"""
            SELECT AVG(systolic), AVG(diastolic)
            FROM blood_pressure_readings
            WHERE patient_id = :patient_id
            AND {date_filter_other}
        """), params).fetchone()
        results["bp"] = f"{int(round(bp[0]))}/{int(round(bp[1]))}" if bp and bp[0] is not None and bp[1] is not None else None

        # TEMPERATURE
        results["temperature"] = db.execute(text(f"""
            SELECT AVG(temperature)
            FROM body_temperature_readings
            WHERE patient_id = :patient_id
            AND {date_filter_other}
        """), params).scalar()

        # SPO2
        results["spo2"] = db.execute(text(f"""
            SELECT AVG(value)
            FROM spo2_readings
            WHERE patient_id = :patient_id
            AND {date_filter_other}
        """), params).scalar()

        # HEART RATE
        results["heart_rate"] = db.execute(text(f"""
            SELECT AVG(value)
            FROM heart_rate_readings
            WHERE patient_id = :patient_id
            AND {date_filter_other}
        """), params).scalar()

        # HRV
        results["hrv"] = db.execute(text(f"""
            SELECT AVG(value)
            FROM hrv_readings
            WHERE patient_id = :patient_id
            AND {date_filter_other}
        """), params).scalar()

        # STRESS
        results["stress"] = db.execute(text(f"""
            SELECT AVG(value)
            FROM stress_readings
            WHERE patient_id = :patient_id
            AND {date_filter_other}
        """), params).scalar()

        
        if target_date:
            # Single day: total sleep minutes for the night ending on target_date
            results["sleep"] = db.execute(text("""
                SELECT COUNT(*)
                FROM sleep_readings_details
                WHERE patient_id = :patient_id
                AND level IN (0, 1, 2)
                AND DATE(actual_time + INTERVAL '12 hours') = :target_date
            """), {"patient_id": patient_id, "target_date": target_date}).scalar()
        elif start_date and end_date:
            # Custom range: average sleep minutes per night between from and to dates
            results["sleep"] = db.execute(text("""
                SELECT
                    CASE
                        WHEN COUNT(DISTINCT DATE(actual_time + INTERVAL '12 hours')) = 0 THEN 0
                        ELSE COUNT(*)::float / COUNT(DISTINCT DATE(actual_time + INTERVAL '12 hours'))
                    END
                FROM sleep_readings_details
                WHERE patient_id = :patient_id
                AND level IN (0, 1, 2)
                AND DATE(actual_time + INTERVAL '12 hours') BETWEEN :start_date AND :end_date
            """), {"patient_id": patient_id, "start_date": start_date, "end_date": end_date}).scalar()
        elif start_date:
            # Weekly: average sleep minutes per night across the window
            results["sleep"] = db.execute(text("""
                SELECT
                    CASE
                        WHEN COUNT(DISTINCT DATE(actual_time + INTERVAL '12 hours')) = 0 THEN 0
                        ELSE COUNT(*)::float / COUNT(DISTINCT DATE(actual_time + INTERVAL '12 hours'))
                    END
                FROM sleep_readings_details
                WHERE patient_id = :patient_id
                AND level IN (0, 1, 2)
                AND DATE(actual_time + INTERVAL '12 hours') >= :start_date
            """), {"patient_id": patient_id, "start_date": start_date}).scalar()
        else:
            # Overall: average sleep minutes per night across all data
            results["sleep"] = db.execute(text("""
                SELECT
                    CASE
                        WHEN COUNT(DISTINCT DATE(actual_time + INTERVAL '12 hours')) = 0 THEN 0
                        ELSE COUNT(*)::float / COUNT(DISTINCT DATE(actual_time + INTERVAL '12 hours'))
                    END
                FROM sleep_readings_details
                WHERE patient_id = :patient_id
                AND level IN (0, 1, 2)
            """), {"patient_id": patient_id}).scalar()

        # ACTIVITY
        # Each row = one activity session (e.g. one Walk), NOT one full day.
        # So for a specific date:  SUM all session rows for that day → correct daily total
        # For overall/weekly:      SUM per day first, THEN AVG those daily totals
        #                          (plain AVG would give avg-per-session, which is wrong)
        if target_date:
            activity = db.execute(text(f"""
                SELECT
                    SUM(total_step),
                    SUM(total_calories_burned),
                    SUM(total_exercise_duration)
                FROM activity_readings
                WHERE patient_id = :patient_id
                AND {date_filter_activity}
            """), params).fetchone()
        else:
            # Subquery: first sum all sessions per day, then average the daily totals
            activity = db.execute(text(f"""
                SELECT
                    AVG(daily_steps),
                    AVG(daily_calories),
                    AVG(daily_duration)
                FROM (
                    SELECT
                        DATE(actual_time)            AS day,
                        SUM(total_step)              AS daily_steps,
                        SUM(total_calories_burned)   AS daily_calories,
                        SUM(total_exercise_duration) AS daily_duration
                    FROM activity_readings
                    WHERE patient_id = :patient_id
                    AND {date_filter_activity}
                    GROUP BY DATE(actual_time)
                ) daily_totals
            """), params).fetchone()

        if activity:
            results["steps"]           = int(activity[0]) if activity[0] is not None else None
            results["calories_burned"] = round(float(activity[1]), 1) if activity[1] is not None else None
            results["active_time"]     = round(float(activity[2]), 1) if activity[2] is not None else None
        else:
            results["steps"]           = None
            results["calories_burned"] = None
            results["active_time"]     = None

        return results

    finally:
        db.close()