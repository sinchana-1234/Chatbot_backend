from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import calendar
import re
from datetime import date, timedelta
from typing import Optional, Tuple
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



# ==========================================================================
# Shared date-range resolution for every patient-data analysis tool.
#
# ONE rule set, ONE place. A tool receives whatever the user said (an explicit
# from/to, a month, a year, or a relative phrase like "last 30 days") and calls
# resolve_range() to get concrete (start, end) 'YYYY-MM-DD' strings plus a human
# label for the response.
#
# Policy: nothing specified -> (None, None, "all available data")  [NO hidden
# 7/14-day default]; explicit date/month/year -> that span; relative/natural
# phrases -> concrete start & end. Tools filter with date_where(); when start/end
# are None the filter is "1=1" (all history).
# ==========================================================================

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})


def _last_day(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _parse_anchor(s: str):
    """A single explicit token -> (span_start, span_end).
    Accepts 'YYYY-MM-DD' (day), 'YYYY-MM' (month), 'YYYY' (year). None otherwise.
    """
    s = s.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        y, m, d = map(int, s.split("-"))
        return date(y, m, d), date(y, m, d)
    if re.fullmatch(r"\d{4}-\d{2}", s):
        y, m = map(int, s.split("-"))
        return date(y, m, 1), date(y, m, _last_day(y, m))
    if re.fullmatch(r"\d{4}", s):
        y = int(s)
        return date(y, 1, 1), date(y, 12, 31)
    return None


def _month_span(year: int, month: int):
    return date(year, month, 1), date(year, month, _last_day(year, month))


def _resolve_period(p: str, today: date):
    """Relative / natural-language phrase -> (start, end). None if unrecognised."""
    p = p.strip().lower()

    m = re.fullmatch(
        r"(?:in |during |over )?(?:the )?last (\d+) "
        r"(day|days|week|weeks|month|months|year|years)", p)
    if m:
        n = int(m.group(1)); unit = m.group(2)
        if unit.startswith("day"):
            return today - timedelta(days=n - 1), today
        if unit.startswith("week"):
            return today - timedelta(days=n * 7 - 1), today
        if unit.startswith("month"):
            total = (today.year * 12 + (today.month - 1)) - (n - 1)
            sy, sm = divmod(total, 12); sm += 1
            return date(sy, sm, 1), today
        if unit.startswith("year"):
            return date(today.year - n + 1, 1, 1), today

    if p == "this week":
        return today - timedelta(days=today.weekday()), today
    if p == "last week":
        this_mon = today - timedelta(days=today.weekday())
        return this_mon - timedelta(days=7), this_mon - timedelta(days=1)
    if p == "this month":
        return _month_span(today.year, today.month)[0], today
    if p == "last month":
        y, mo = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
        return _month_span(y, mo)
    if p == "this year":
        return date(today.year, 1, 1), today
    if p == "last year":
        return date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)

    if p in ("recently", "recent"):
        return today - timedelta(days=29), today          # convention: last 30 days

    m = re.fullmatch(r"(?:in |during )?([a-z]+)(?: (\d{4}))?", p)
    if m and m.group(1) in _MONTHS:
        mo = _MONTHS[m.group(1)]; y = int(m.group(2)) if m.group(2) else today.year
        return _month_span(y, mo)

    m = re.fullmatch(r"since ([a-z]+)(?: (\d{4}))?", p)
    if m and m.group(1) in _MONTHS:
        mo = _MONTHS[m.group(1)]; y = int(m.group(2)) if m.group(2) else today.year
        return date(y, mo, 1), today

    m = re.fullmatch(r"(?:from )?([a-z]+) (?:to|until|through|-) ([a-z]+)(?: (\d{4}))?", p)
    if m and m.group(1) in _MONTHS and m.group(2) in _MONTHS:
        y = int(m.group(3)) if m.group(3) else today.year
        m1, m2 = _MONTHS[m.group(1)], _MONTHS[m.group(2)]
        return date(y, m1, 1), date(y, m2, _last_day(y, m2))

    return None


def _label(start, end) -> str:
    """Human-readable span for the response. Windows-safe (no %-d)."""
    if not start and not end:
        return "all available data"
    try:
        s = start.strftime("%B ") + str(start.day) + f", {start.year}"
        e = end.strftime("%B ") + str(end.day) + f", {end.year}"
    except AttributeError:
        return "all available data"
    return f"{s} to {e}"


def resolve_range(from_date=None, to_date=None, period=None, today=None):
    """
    -> (start_iso, end_iso, label). start/end are 'YYYY-MM-DD' or None (all history).
    Precedence: explicit from/to  >  period phrase  >  nothing.
    """
    today = today or date.today()

    if from_date or to_date:
        a = _parse_anchor(from_date) if from_date else None
        b = _parse_anchor(to_date) if to_date else None
        if from_date and to_date and a and b:
            start, end = a[0], b[1]
        elif from_date and a:
            start, end = a
            if start == end:            # a lone single DAY = "from that day onward"
                end = today             # (a 1-day trend is never intended)
        elif to_date and b:
            start, end = b
        else:
            return None, None, "all available data"     # unparseable -> safe default
        return start.isoformat(), end.isoformat(), _label(start, end)

    if period:
        r = _resolve_period(period, today)
        if r:
            start, end = r
            return start.isoformat(), end.isoformat(), _label(start, end)

    return None, None, "all available data"


def date_where(column: str, start, end, prefix: str = "dt"):
    """
    Build a Postgres date filter for a timestamp column.
    Returns (sql_fragment, params). '1=1' when no bounds (all history).
    """
    clauses, params = [], {}
    if start:
        clauses.append(f"DATE({column}) >= :{prefix}_start")
        params[f"{prefix}_start"] = start
    if end:
        clauses.append(f"DATE({column}) <= :{prefix}_end")
        params[f"{prefix}_end"] = end
    return (" AND ".join(clauses) if clauses else "1=1"), params