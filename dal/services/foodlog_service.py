#!/usr/bin/env python3
"""
Food log service for handling food log and nutrition data
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from .base_service import BaseService

logger = logging.getLogger(__name__)


def _num(x):
    """Always return a plain number or None. analysis_data is not uniform: a value
    like best_estimate or grams is sometimes the number itself, sometimes wrapped
    as {"value": 55}. Flatten either shape so downstream (and the UI) only ever see
    a number — an object here crashes the React tooltip."""
    if isinstance(x, dict):
        x = x.get("value", x.get("grams", x.get("best_estimate")))
    return x if isinstance(x, (int, float)) else None

def _foodlog_nutrition(analysis_data):
    """Parse calories + macro grams from analysis_data JSON. Real format nests
    macros under macronutrients.<macro>.grams. None for anything not present."""
    out = {"calories": None, "carbs_g": None, "protein_g": None, "fat_g": None}
    if not analysis_data or str(analysis_data).strip().lower() in ("null", ""):
        return out
    try:
        d = json.loads(analysis_data)
        out["calories"] = _num((d.get("total_calories") or {}).get("best_estimate"))
        macros = d.get("macronutrients") or {}
        for key, field in (("carbohydrates", "carbs_g"), ("protein", "protein_g"), ("fats", "fat_g")):
            m = macros.get(key)
            out[field] = _num(m.get("grams") if isinstance(m, dict) else m)
    except (ValueError, TypeError, AttributeError):
        pass
    return out

class FoodlogService(BaseService):
    """Service for handling food log operations"""
    
    def __init__(self, db_session: Session):
        super().__init__(db_session)
    
    def get_foodlog(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
                   date_filter: Optional[datetime] = None,
                   from_date: Optional[str] = None, to_date: Optional[str] = None,
                   period: Optional[str] = None,
                   limit: Optional[int] = 10) -> Dict[str, Any]:
        """Get food log records for a patient.

        Patient resolution stays on MySQL (the `users` table is MySQL-only), but the
        foodlog rows are read from POSTGRES: public.foodlog is the source of truth and
        is the only copy carrying actual_time / meal_type / analysis_data. Uses the same
        SessionLocalPG the glucose tools use.
        """
        try:
            from sqlalchemy import text
            from dal.postgres_db import SessionLocalPG, resolve_range, date_where
            from dal.cycle_window import cycle_window

            # Resolve the patient on MySQL (users lives there)
            patient_id = self.find_patient_by_name_or_id(patient_id, patient_name)
            if not patient_id:
                return {"error": "Patient not found"}

            # Resolve an optional date range. Range wins over the legacy date_filter.
            start = end = period_label = None
            if from_date or to_date or period:
                start, end, period_label = resolve_range(from_date, to_date, period)
                if not start:
                    # A period phrase like "cycle" that resolve_range can't date on
                    # its own -> fall back to the patient's current program cycle.
                    cw = cycle_window(patient_id)
                    if cw:
                        start, end = cw
                        period_label = f"{start} to {end}"
            range_active = bool(start)

            sql = (
                "SELECT id, type, url, activitydate, createdon, actual_time, "
                "source_timezone, createdby, description, status, latitude, "
                "longitude, analysis_data, meal_type "
                "FROM foodlog "
                "WHERE patient_id = :pid AND status = 1"
            )
            params = {"pid": patient_id}
            if range_active:
                # DATE(actual_time) BETWEEN start AND end, both inclusive.
                frag, dparams = date_where("actual_time", start, end)
                sql += f" AND {frag}"
                params.update(dparams)
            elif date_filter:
                sql += " AND actual_time >= :dfrom"
                params["dfrom"] = date_filter
            sql += " ORDER BY actual_time DESC NULLS LAST"

            # A range wants the whole window; the plain "latest" query still caps.
            eff_limit = None if range_active else limit
            if eff_limit is not None:
                sql += " LIMIT :lim"
                params["lim"] = eff_limit

            pg = SessionLocalPG()
            try:
                rows = pg.execute(text(sql), params).mappings().all()
            finally:
                pg.close()

            foodlog_list = []
            for r in rows:
                nut = _foodlog_nutrition(r["analysis_data"])
                foodlog_list.append({
                    "id": r["id"],
                    "type": r["type"],
                    "url": r["url"],
                    "activitydate": r["activitydate"],
                    "createdon": r["createdon"].isoformat() if r["createdon"] is not None else None,
                    "actual_time": r["actual_time"].isoformat() if r["actual_time"] is not None else None,
                    "source_timezone": r["source_timezone"],
                    "createdby": r["createdby"],
                    "description": r["description"],
                    "status": r["status"],
                    "latitude": r["latitude"],
                    "longitude": r["longitude"],
                    "analysis_data": r["analysis_data"],
                    "calories": nut["calories"],
                    "carbs_g": nut["carbs_g"],
                    "protein_g": nut["protein_g"],
                    "fat_g": nut["fat_g"],
                    "meal_type": r["meal_type"],
                })

            if range_active:
                message = (f"Showing {len(foodlog_list)} foodlog records "
                           f"from {start} to {end}")
            elif date_filter:
                message = (f"Showing top {len(foodlog_list)} latest foodlog records "
                           f"from {date_filter.strftime('%Y-%m-%d')}")
            else:
                message = f"Showing top {len(foodlog_list)} latest foodlog records"

            return {
                "patient_id": patient_id,
                "foodlog": foodlog_list,
                "count": len(foodlog_list),
                "limit_applied": eff_limit,
                "date_filter": date_filter.isoformat() if date_filter else None,
                "date_from": start,
                "date_to": end,
                "period_label": period_label,
                "message": message,
            }

        except Exception as e:
            logger.error(f"Error getting foodlog: {e}")
            return {"error": f"Database error: {str(e)}"}

    def get_foodlog_uploaders_by_date(self, date_filter: datetime,
                                       roster_patient_ids: Optional[List[int]] = None,
                                       include_items: bool = False) -> Dict[str, Any]:
        """Every patient who uploaded a food log on a given LOCAL date.

        Foodlog rows come from POSTGRES; names are resolved from MySQL users afterwards
        (the old single-query JOIN across Foodlog+Users is impossible once foodlog is in
        a different database). Date is matched on actual_time (patient-local), matching
        the local-time policy used everywhere else in the Postgres layer.
        """
        try:
            from sqlalchemy import text
            from dal.postgres_db import SessionLocalPG
            from ..models.users import Users

            target = date_filter.strftime("%Y-%m-%d")

            if roster_patient_ids is not None and not roster_patient_ids:
                return {"date": target, "count": 0, "uploaders": [],
                        "message": f"No patients in scope to check for {target}."}

            where = ["status = 1", "actual_time IS NOT NULL", "DATE(actual_time) = :target"]
            params = {"target": target}
            if roster_patient_ids is not None:
                where.append("patient_id = ANY(:roster)")
                params["roster"] = roster_patient_ids
            where_sql = " AND ".join(where)

            pg = SessionLocalPG()
            try:
                if not include_items:
                    rows = pg.execute(text(
                        f"SELECT DISTINCT patient_id FROM foodlog WHERE {where_sql}"
                    ), params).mappings().all()
                    pid_order = [r["patient_id"] for r in rows]
                    items_by_pid = {}
                else:
                    rows = pg.execute(text(
                        f"SELECT patient_id, description, url, type, meal_type, actual_time ,  analysis_data "
                        f"FROM foodlog WHERE {where_sql} ORDER BY patient_id, actual_time ASC"
                    ), params).mappings().all()
                    items_by_pid, pid_order = {}, []
                    for r in rows:
                        pid = r["patient_id"]
                        if pid not in items_by_pid:
                            items_by_pid[pid] = []
                            pid_order.append(pid)
                        nut = _foodlog_nutrition(r["analysis_data"])
                        items_by_pid[pid].append({
                            "time": r["actual_time"].strftime("%I:%M %p") if r["actual_time"] else None,
                            "description": r["description"], "url": r["url"],
                            "type": r["type"], "meal_type": r["meal_type"],
                            "calories": nut["calories"], "carbs_g": nut["carbs_g"],
                            "protein_g": nut["protein_g"], "fat_g": nut["fat_g"],
                        })
            finally:
                pg.close()

            if not pid_order:
                return {"date": target, "count": 0, "uploaders": [],
                        "message": f"No patients uploaded a food log on {target}."}

            # Resolve names on MySQL for just these patient IDs
            users = self.db.query(Users).filter(Users.id.in_(pid_order)).all()
            name_by_id = {u.id: (f"{u.first_name or ''} {u.last_name or ''}".strip()) for u in users}

            uploaders = []
            for pid in pid_order:
                entry = {"patient_name": name_by_id.get(pid) or f"Patient {pid}"}
                if include_items:
                    entry["items"] = items_by_pid.get(pid, [])
                uploaders.append(entry)
            uploaders.sort(key=lambda u: u["patient_name"].lower())

            return {"date": target, "count": len(uploaders), "uploaders": uploaders,
                    "message": f"{len(uploaders)} patient(s) uploaded a food log on {target}."}

        except Exception as e:
            logger.error(f"Error getting foodlog uploaders by date: {e}")
            return {"error": f"Database error: {str(e)}"}