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
        """Get food log records for a patient"""
        try:
            from ..models.foodlog import Foodlog
            
            # Find patient ID
            patient_id = self.find_patient_by_name_or_id(patient_id, patient_name)
            if not patient_id:
                return {"error": "Patient not found"}
            
            # Get active foodlog records (status = 1)
            query = self.db.query(Foodlog).filter(
                Foodlog.patient_id == patient_id,
                Foodlog.status == 1
            )
            
            # Apply date filter if provided (on createdon)
            if date_filter:
                query = query.filter(Foodlog.createdon >= date_filter)
            
            # Order by createdon descending and limit results
            query = query.order_by(Foodlog.createdon.desc()).limit(limit)
            foodlogs = query.all()
            
            # Convert to dict
            foodlog_list = []
            for log in foodlogs:
                log_dict = {
                    "id": log.id,
                    "type": log.type,
                    "url": log.url,
                    "activitydate": log.activitydate,
                    "createdon": log.createdon.isoformat() if log.createdon is not None else None,
                    "createdby": log.createdby,
                    "description": log.description,
                    "status": log.status,
                    "latitude": log.latitude,
                    "longitude": log.longitude
                }
                foodlog_list.append(log_dict)
            
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
        """
        Get every patient who uploaded at least one food log on a given date.

        Args:
            date_filter (datetime): The date to check (time portion is ignored;
                the whole day, start to end, is checked).
            roster_patient_ids (List[int], optional): Restrict results to this
                set of patient IDs (e.g. a doctor's own patients). If None,
                all patients in the system are considered.
            include_items (bool): If True, also include each patient's actual
                food log entries (description/url/time) for that date, not
                just their name. Use when the question asks what was eaten,
                not just who uploaded.

        Returns:
            dict: Resolved date, count, and list of uploaders. Each uploader
                has "patient_name", and "items" (list of that patient's
                entries) when include_items is True.
        """
        try:
            from datetime import timedelta
            from ..models.foodlog import Foodlog
            from ..models.users import Users

            day_start = date_filter.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)

            if not include_items:
                # Lightweight path: just who uploaded, one row per patient
                query = self.db.query(
                    Foodlog.patient_id,
                    Users.first_name,
                    Users.last_name
                ).join(
                    Users, Foodlog.patient_id == Users.id
                ).filter(
                    Foodlog.status == 1,
                    Foodlog.createdon >= day_start,
                    Foodlog.createdon < day_end
                )

                if roster_patient_ids is not None:
                    if not roster_patient_ids:
                        return {
                            "date": day_start.strftime("%Y-%m-%d"),
                            "count": 0,
                            "uploaders": [],
                            "message": f"No patients in scope to check for {day_start.strftime('%Y-%m-%d')}."
                        }
                    query = query.filter(Foodlog.patient_id.in_(roster_patient_ids))

                # DISTINCT on patient_id so multiple uploads by the same patient
                # on the same day only count once. Grouping by all three selected
                # columns (rather than dialect-specific DISTINCT ON) keeps this
                # portable across MySQL and Postgres.
                results = query.group_by(
                    Foodlog.patient_id, Users.first_name, Users.last_name
                ).all()

                uploaders = []
                for patient_id, first_name, last_name in results:
                    full_name = f"{first_name or ''} {last_name or ''}".strip() or f"Patient {patient_id}"
                    uploaders.append({"patient_name": full_name})

                uploaders.sort(key=lambda u: u["patient_name"].lower())

                date_str = day_start.strftime("%Y-%m-%d")
                return {
                    "date": date_str,
                    "count": len(uploaders),
                    "uploaders": uploaders,
                    "message": f"{len(uploaders)} patient(s) uploaded a food log on {date_str}."
                }

            # include_items path: fetch every matching row (not collapsed),
            # then group each patient's entries together in Python
            query = self.db.query(
                Foodlog.patient_id,
                Users.first_name,
                Users.last_name,
                Foodlog.description,
                Foodlog.url,
                Foodlog.type,
                Foodlog.createdon
            ).join(
                Users, Foodlog.patient_id == Users.id
            ).filter(
                Foodlog.status == 1,
                Foodlog.createdon >= day_start,
                Foodlog.createdon < day_end
            )

            if roster_patient_ids is not None:
                if not roster_patient_ids:
                    return {
                        "date": day_start.strftime("%Y-%m-%d"),
                        "count": 0,
                        "uploaders": [],
                        "message": f"No patients in scope to check for {day_start.strftime('%Y-%m-%d')}."
                    }
                query = query.filter(Foodlog.patient_id.in_(roster_patient_ids))

            rows = query.order_by(Foodlog.patient_id, Foodlog.createdon.asc()).all()

            patients_by_id: Dict[int, Dict[str, Any]] = {}
            for patient_id, first_name, last_name, description, url, log_type, createdon in rows:
                if patient_id not in patients_by_id:
                    full_name = f"{first_name or ''} {last_name or ''}".strip() or f"Patient {patient_id}"
                    patients_by_id[patient_id] = {"patient_name": full_name, "items": []}
                patients_by_id[patient_id]["items"].append({
                    "time": createdon.strftime("%I:%M %p") if createdon else None,
                    "description": description,
                    "url": url,
                    "type": log_type
                })

            uploaders = sorted(patients_by_id.values(), key=lambda u: u["patient_name"].lower())

            date_str = day_start.strftime("%Y-%m-%d")
            return {
                "date": date_str,
                "count": len(uploaders),
                "uploaders": uploaders,
                "message": f"{len(uploaders)} patient(s) uploaded a food log on {date_str}."
            }

        except Exception as e:
            logger.error(f"Error getting foodlog uploaders by date: {e}")
            return {"error": f"Database error: {str(e)}"}