from sqlalchemy import Column, Integer, String

from .base import Base

class Role(Base):
    __tablename__ = "role"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    
    # Role mapping based on your database
    PATIENT = 1
    DOCTOR = 2
    HEALTH_COACH = 3
    ADMIN = 4
    DIAGNOSTIC = 5
    VIDEO_UPLOADER = 6
    TRAINER = 7
    TRACKER = 8
    CRM_ADMIN = 9
    CRM_EXECUTIVE = 10
    VENDOR = 11
    ORDER_MANAGER = 12
    VIDEO_ADMIN = 13
    READ_ONLY = 14
