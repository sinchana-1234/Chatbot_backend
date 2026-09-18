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
