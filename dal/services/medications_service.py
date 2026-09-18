#!/usr/bin/env python3
"""
Medications service for handling medication and supplement data
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from .base_service import BaseService

logger = logging.getLogger(__name__)

class MedicationsService(BaseService):
    """Service for handling medications operations"""
    
    def __init__(self, db_session: Session):
        super().__init__(db_session)
    
    def get_medications(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
                       date_filter: Optional[datetime] = None, limit: int = 10) -> Dict[str, Any]:
        """Get current medications for a patient"""
        try:
            from ..models.medications import Medications
            
            # Find patient ID
            patient_id = self.find_patient_by_name_or_id(patient_id, patient_name)
            if not patient_id:
                return {"error": "Patient not found"}
            
            # Get active medications (status = 1)
            query = self.db.query(Medications).filter(
                Medications.patient_id == patient_id,
                Medications.status == 1
            )
            
            # Apply date filter if provided
            if date_filter:
                query = query.filter(Medications.created >= date_filter)
            
            # Order by created date descending and limit results
            query = query.order_by(Medications.created.desc()).limit(limit)
            medications = query.all()
            
            # Convert to dict
            medication_list = []
            for med in medications:
                medication_dict = {
                    "id": med.id,
                    "medication_type": med.medication_type,
                    "medication_name": med.medication_name,
                    "dosage": med.dosage,
                    "frequency": med.frequency,
                    "start_date": med.start_date.isoformat() if med.start_date is not None else None,
                    "end_date": med.end_date.isoformat() if med.end_date is not None else None,
                    "note": med.note,
                    "progress": med.progress,
                    "created": med.created.isoformat() if med.created is not None else None
                }
                medication_list.append(medication_dict)
            
            return {
                "patient_id": patient_id,
                "medications": medication_list,
                "count": len(medication_list),
                "limit_applied": limit,
                "date_filter": date_filter.isoformat() if date_filter else None,
                "message": f"Showing top {len(medication_list)} latest medications" + (f" from {date_filter.strftime('%Y-%m-%d')}" if date_filter else "")
            }
            
        except Exception as e:
            logger.error(f"Error getting medications: {e}")
            return {"error": f"Database error: {str(e)}"}
