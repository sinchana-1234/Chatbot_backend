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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )


settings = Settings()