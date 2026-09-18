#!/usr/bin/env python3
"""
Base service class for medical data access
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class BaseService:
    """Base service class with common database operations"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def find_patient_by_name_or_id(self, patient_id: Optional[int] = None, 
                                  patient_name: Optional[str] = None):
        """Find patient ID from name or ID"""
        from ..models.users import Users
        
        if patient_name and not patient_id:
            user_query = self.db.query(Users)
            name_lower = patient_name.lower().strip()
            
            # Split the name into parts for better matching
            name_parts = name_lower.split()
            
            if len(name_parts) >= 2:
                # If we have multiple parts, try to match first + last name combination
                first_part = name_parts[0]
                last_part = name_parts[-1]  # Use last part as last name
                
                users = user_query.filter(
                    (Users.first_name.ilike(f"%{first_part}%")) &
                    (Users.last_name.ilike(f"%{last_part}%"))
                ).all()
                
                # If no exact match, try broader search
                if not users:
                    users = user_query.filter(
                        (Users.first_name.ilike(f"%{name_lower}%")) |
                        (Users.last_name.ilike(f"%{name_lower}%")) |
                        ((Users.first_name + ' ' + Users.last_name).ilike(f"%{name_lower}%"))
                    ).all()
            else:
                # Single name, search in both first and last name
                users = user_query.filter(
                    (Users.first_name.ilike(f"%{name_lower}%")) |
                    (Users.last_name.ilike(f"%{name_lower}%"))
                ).all()
                
            if users:
                return getattr(users[0], 'id')
        
        return patient_id
    
    def get_user_info(self, patient_id: int):
        """Get user information by ID"""
        from ..models.users import Users
        
        user = self.db.query(Users).filter(Users.id == patient_id).first()
        if user:
            return {
                "id": user.id,
                "name": f"{user.first_name} {user.last_name}",
                "mobile": user.mobile_number,
                "email": user.email
            }
        return None
    
    def apply_date_filter(self, query, model, field_name: str, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None):
        """Apply date filtering to a query"""
        if start_date:
            query = query.filter(getattr(model, field_name) >= start_date)
        if end_date:
            query = query.filter(getattr(model, field_name) <= end_date)
        return query
