#!/usr/bin/env python3
"""
Food log service for handling food log and nutrition data
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from .base_service import BaseService

logger = logging.getLogger(__name__)

class FoodlogService(BaseService):
    """Service for handling food log operations"""
    
    def __init__(self, db_session: Session):
        super().__init__(db_session)
    
    def get_foodlog(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
                   date_filter: Optional[datetime] = None, limit: int = 10) -> Dict[str, Any]:
        """Get food log records for a patient.

        Patient resolution stays on MySQL (the `users` table is MySQL-only), but the
        foodlog rows are read from POSTGRES: public.foodlog is the source of truth and
        is the only copy carrying actual_time / meal_type / analysis_data. Uses the same
        SessionLocalPG the glucose tools use.
        """
        try:
            from sqlalchemy import text
            from dal.postgres_db import SessionLocalPG

            # Resolve the patient on MySQL (users lives there)
            patient_id = self.find_patient_by_name_or_id(patient_id, patient_name)
            if not patient_id:
                return {"error": "Patient not found"}

            sql = (
                "SELECT id, type, url, activitydate, createdon, actual_time, "
                "source_timezone, createdby, description, status, latitude, "
                "longitude, analysis_data, meal_type "
                "FROM foodlog "
                "WHERE patient_id = :pid AND status = 1"
            )
            params = {"pid": patient_id, "lim": limit}
            if date_filter:
                sql += " AND actual_time >= :dfrom"
                params["dfrom"] = date_filter
            sql += " ORDER BY actual_time DESC NULLS LAST LIMIT :lim"

            pg = SessionLocalPG()
            try:
                rows = pg.execute(text(sql), params).mappings().all()
            finally:
                pg.close()

            foodlog_list = []
            for r in rows:
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
                    "meal_type": r["meal_type"],
                })

            return {
                "patient_id": patient_id,
                "foodlog": foodlog_list,
                "count": len(foodlog_list),
                "limit_applied": limit,
                "date_filter": date_filter.isoformat() if date_filter else None,
                "message": f"Showing top {len(foodlog_list)} latest foodlog records" + (f" from {date_filter.strftime('%Y-%m-%d')}" if date_filter else "")
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
                        f"SELECT patient_id, description, url, type, meal_type, actual_time "
                        f"FROM foodlog WHERE {where_sql} ORDER BY patient_id, actual_time ASC"
                    ), params).mappings().all()
                    items_by_pid, pid_order = {}, []
                    for r in rows:
                        pid = r["patient_id"]
                        if pid not in items_by_pid:
                            items_by_pid[pid] = []
                            pid_order.append(pid)
                        items_by_pid[pid].append({
                            "time": r["actual_time"].strftime("%I:%M %p") if r["actual_time"] else None,
                            "description": r["description"], "url": r["url"],
                            "type": r["type"], "meal_type": r["meal_type"],
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