#!/usr/bin/env python3
"""
Protocol service for handling medical protocols and treatment plans
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from .base_service import BaseService

logger = logging.getLogger(__name__)

class ProtocolService(BaseService):
    """Service for handling protocol operations"""
    
    def __init__(self, db_session: Session):
        super().__init__(db_session)
    
    def get_protocols(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
                     date_filter: Optional[datetime] = None, limit: int = 10) -> Dict[str, Any]:
        """Get protocol records for a patient"""
        try:
            from ..models.protocol import Protocol
            
            # Find patient ID
            patient_id = self.find_patient_by_name_or_id(patient_id, patient_name)
            if not patient_id:
                return {"error": "Patient not found"}
            
            # Get active protocol records (status = 1)
            query = self.db.query(Protocol).filter(
                Protocol.patient_id == patient_id,
                Protocol.status == 1
            )
            
            # Apply date filter if provided (on createdon)
            if date_filter:
                query = query.filter(Protocol.createdon >= date_filter)
            
            # Order by createdon descending and limit results
            query = query.order_by(Protocol.createdon.desc()).limit(limit)
            protocols = query.all()
            
            # Convert to dict
            protocol_list = []
            for protocol in protocols:
                protocol_dict = {
                    "id": protocol.id,
                    "doctor_id": protocol.doctor_id,
                    "patient_id": protocol.patient_id,
                    "createdon": protocol.createdon.isoformat() if protocol.createdon is not None else None,
                    "createdby": protocol.createdby,
                    "status": protocol.status,
                    "description": protocol.description  # Full HTML formatted protocol content
                }
                protocol_list.append(protocol_dict)
            
            return {
                "patient_id": patient_id,
                "protocols": protocol_list,
                "count": len(protocol_list),
                "limit_applied": limit,
                "date_filter": date_filter.isoformat() if date_filter else None,
                "message": f"Showing top {len(protocol_list)} latest protocol records" + (f" from {date_filter.strftime('%Y-%m-%d')}" if date_filter else "")
            }
            
        except Exception as e:
            logger.error(f"Error getting protocols: {e}")
            return {"error": f"Database error: {str(e)}"}
