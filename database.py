from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String, Index
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from config import settings
from typing import Generator


# ─────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────
# Apply SSL only for managed cloud Postgres (Azure, AWS RDS, GCP).
# Local/QA Postgres at internal IPs typically doesn't support SSL —
# applying these args there breaks the connection.
_db_url = settings.DATABASE_URL or ""
_needs_ssl = any(host in _db_url for host in [
    ".database.azure.com",
    ".rds.amazonaws.com",
    ".gcp.cloud.sql",
])
_connect_args = {
    "sslmode": "require",
    "sslrootcert": "/certs/DigiCertGlobalRootG2.crt.pem",
} if _needs_ssl else {}

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    future=True,
    connect_args=_connect_args,
)


# ─────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


Base = declarative_base()


# ─────────────────────────────────────────────
# ORM Model
# ─────────────────────────────────────────────
class CGMReading(Base):
    """
    CGM minute-level glucose readings.
    """

    __tablename__ = "glucose_readings"

    id = Column(Integer, primary_key=True, index=True)

    # Patient reference
    patient_id = Column(Integer, nullable=False, index=True)

    # Glucose value
    glucose = Column("glucose_value", Float, nullable=False)

    # Time columns
    event_time_utc = Column(DateTime, nullable=False)
    local_event_time = Column(DateTime, nullable=False)

    # Metadata
    source_timezone = Column(String)
    created_at = Column(DateTime)

    # Composite index for faster queries
    __table_args__ = (
        Index("idx_patient_local_time", "patient_id", "local_event_time"),
    )


# ─────────────────────────────────────────────
# DB Dependency for FastAPI
# ─────────────────────────────────────────────
def get_db() -> Generator[Session, None, None]:

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────────
# Create Tables (Development Only)
# ─────────────────────────────────────────────
def create_tables():

    Base.metadata.create_all(bind=engine)