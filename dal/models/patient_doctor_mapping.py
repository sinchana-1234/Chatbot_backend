#!/usr/bin/env python3
"""
Patient Doctor Mapping Model - Revival Medical System
"""

from sqlalchemy import Column, Integer, DateTime, SmallInteger, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base

class PatientDoctorMapping(Base):
    """Patient Doctor Mapping table model"""
    __tablename__ = 'patients_doctors_mapping'
    
    # Composite primary key
    user_id = Column(Integer, primary_key=True, nullable=False)
    patient_id = Column(Integer, primary_key=True, nullable=False)
    
    # Date fields
    from_date = Column(DateTime, nullable=True)
    to_date = Column(DateTime, nullable=True)
    
    # Primary doctor flag
    is_primary = Column(SmallInteger, nullable=False, default=0)
    
    def __repr__(self):
        return f"<PatientDoctorMapping(user_id={self.user_id}, patient_id={self.patient_id}, is_primary={self.is_primary})>"
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'user_id': self.user_id,
            'patient_id': self.patient_id,
            'from_date': self.from_date.isoformat() if self.from_date else None,
            'to_date': self.to_date.isoformat() if self.to_date else None,
            'is_primary': bool(self.is_primary)
        }
    
    @property
    def is_primary_doctor(self):
        """Check if this is a primary doctor mapping"""
        return bool(self.is_primary)
    
    @property
    def is_active(self):
        """Check if the mapping is currently active"""
        from datetime import datetime
        now = datetime.now()
        
        # If from_date is set and in the future, not active yet
        if self.from_date and self.from_date > now:
            return False
        
        # If to_date is set and in the past, no longer active
        if self.to_date and self.to_date < now:
            return False
        
        # Active if from_date is past/now and to_date is future/None
        return True
