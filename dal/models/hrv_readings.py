from sqlalchemy import Column, Integer, Float, DateTime

from .base import Base

class HrvReadings(Base):
    __tablename__ = "hrv_readings"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=True)
    value = Column(Float, nullable=True)
    patient_id = Column(Integer, nullable=True)
    actual_time = Column(DateTime, nullable=True)
