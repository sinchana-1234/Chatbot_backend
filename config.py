from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database — MUST come from .env, no default so missing config fails loudly
    DATABASE_URL: str

    # App
    APP_NAME: str = "CGM Glucose Predictive Analysis"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # External APIs
    HEALTH_PROGRESS_BASE_URL: str  # must be set in .env — no default so missing config fails loudly

    # Model
    RF_N_ESTIMATORS: int = 100  # reduced from 200 — on a 2-core box 200 trees doubles CPU time per training with minimal accuracy gain
    RF_RANDOM_STATE: int = 42
    MIN_WEEKS_FOR_PREDICTION: int = 2   # enforce: predict only if ≥2 weeks of data

    # Activity → glucose correlation analysis knobs
    LOW_ACTIVITY_MAX_STEPS: int = 4000       # steps < this  → "low activity" day
    HIGH_ACTIVITY_MIN_STEPS: int = 8000      # steps > this  → "high activity" day
    MIN_DAYS_TO_COMPARE: int = 5             # need this many days with BOTH activity+glucose before drawing any conclusion
    MIN_DAYS_FOR_FIRM_RESULT: int = 7        # a compared band with fewer days than this is flagged "suggestive"
    MIN_GLUCOSE_DIFFERENCE_MGDL: int = 5     # glucose gap below this  → "no clear effect"

    # Sleep → glucose (reuses MIN_DAYS_TO_COMPARE / MIN_DAYS_FOR_FIRM_RESULT / MIN_GLUCOSE_DIFFERENCE_MGDL)
    SLEEP_SHORT_MAX_HOURS: int = 6           # < this  → "short sleep" night
    SLEEP_LONG_MIN_HOURS: int = 8            # > this  → "long sleep" night

    # Stress → glucose (reuses MIN_DAYS_TO_COMPARE / MIN_GLUCOSE_DIFFERENCE_MGDL)
    STRESS_THIN_SIDE_DAYS: int = 5           # a side with fewer days than this is flagged "interpret with caution"

    # Meal → glucose impact
    MEAL_POSTMEAL_WINDOW_HOURS: int = 2      # postprandial window measured after each meal (standard)
    MEAL_BASELINE_TOLERANCE_MIN: int = 30    # reading nearest the meal within +/- this = baseline
    MEAL_MIN_MEALS_FOR_RANKING: int = 5      # fewer usable meals than this = "small sample", not a ranking

    # Health-progress trend tools. Their external API is a /{from}/{to} path and
    # can't take "unbounded", so "all history" (no period given) is realised as
    # this many days back from today. The response states the actual dates.
    TREND_ALL_HISTORY_LOOKBACK_DAYS: int = 730   # ~2 years

    # AGP chart. AGP is a clinical report defined over a BOUNDED window (a 24-hour
    # ribbon built by overlaying the days in range), so — unlike the trend and
    # correlation tools — it does NOT default to all-history. With no period given
    # it uses this window; an explicit period still overrides it.
    AGP_DEFAULT_WINDOW_DAYS: int = 14

    # Glucose interpretation thresholds (mg/dL) for the patient-summary tool.
    # Single source of truth — the tool reads these instead of hardcoding numbers.
    GLUCOSE_LOW_MGDL: int = 70        # < this  → Low
    GLUCOSE_ELEVATED_MGDL: int = 140  # > this  → slightly elevated (top of "normal")
    GLUCOSE_HIGH_MGDL: int = 180      # > this  → High

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )


settings = Settings()