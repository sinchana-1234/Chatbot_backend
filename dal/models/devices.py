#!/usr/bin/env python3
"""
Devices Model - Revival Medical System
SQLAlchemy model based on actual database schema
"""

from sqlalchemy import Column, Integer, String, DateTime, SmallInteger
from datetime import datetime, timedelta
from .base import Base

class Devices(Base):
    """Devices table model for CGM and other medical devices"""
    __tablename__ = 'devices'
    
    # Based on actual database schema from DESCRIBE devices
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=True)  # Device name like "CGM", "BLE-J2208A 05D968"
    tag_id = Column(String(255), nullable=True)  # Device tag identifier
    mapped_date = Column(DateTime, nullable=True)  # When device was mapped
    patient_id = Column(Integer, nullable=False)  # Patient ID
    status = Column(SmallInteger, nullable=True, default=1)  # 1=Active, 2=Inactive
    session_start_date = Column(DateTime, nullable=True)  # When device session started
    
    def __repr__(self):
        return f"<Devices(id={self.id}, patient_id={self.patient_id}, name='{self.name}', status={self.status})>"
    
    def to_dict(self):
        """Convert device object to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'tag_id': self.tag_id,
            'mapped_date': self.mapped_date.isoformat() if self.mapped_date else None,
            'patient_id': self.patient_id,
            'status': self.status,
            'session_start_date': self.session_start_date.isoformat() if self.session_start_date else None
        }
    
    @property
    def is_active(self):
        """Check if the device is currently active"""
        return self.status == 1
    
    @property 
    def is_expired(self):
        """Check if the device session has expired (for CGM: session_start_date + 15 days < today)"""
        if not self.session_start_date or not self.is_active:
            return True
            
        # For CGM devices, they expire 15 days after session start
        if self.name and 'cgm' in self.name.lower():
            expiry_date = self.session_start_date + timedelta(days=15)
            return expiry_date < datetime.now()
        
        # For other devices, assume they don't expire unless specified
        return False
    
    @property
    def expiry_date(self):
        """Get the expiry date for the device"""
        if not self.session_start_date:
            return None
            
        # For CGM devices, they expire 15 days after session start
        if self.name and 'cgm' in self.name.lower():
            return self.session_start_date + timedelta(days=15)
        
        # For other devices, no expiry date
        return None
    
    @property
    def days_until_expiry(self):
        """Get number of days until device expires"""
        expiry = self.expiry_date
        if not expiry:
            return None
            
        delta = expiry - datetime.now()
        return delta.days if delta.days >= 0 else 0  # Return 0 if already expired
