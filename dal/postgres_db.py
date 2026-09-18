from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

POSTGRES_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:"
    f"{os.getenv('POSTGRES_PORT')}/"
    f"{os.getenv('POSTGRES_DB')}"
) if all([os.getenv('POSTGRES_USER'), os.getenv('POSTGRES_HOST')]) else os.getenv("DATABASE_URL")

# Apply SSL only for managed cloud Postgres (Azure, AWS RDS, GCP).
# Local/QA Postgres at internal IPs (e.g. 10.0.0.15) typically doesn't
# support SSL — applying these args there breaks the connection.
# This mirrors the logic in /database.py so all three Postgres engines
# behave the same way against cloud vs internal DBs.
_needs_ssl = any(host in (POSTGRES_URL or "") for host in [
    ".database.azure.com",
    ".rds.amazonaws.com",
    ".gcp.cloud.sql",
])
_connect_args = {
    "sslmode": "require",
    "sslrootcert": "/certs/DigiCertGlobalRootG2.crt.pem",
} if _needs_ssl else {}

engine_pg = create_engine(
    POSTGRES_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    connect_args=_connect_args,
) if POSTGRES_URL else None

SessionLocalPG = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine_pg
) if engine_pg else None