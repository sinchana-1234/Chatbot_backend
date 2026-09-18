from sqlalchemy import Column, Integer, DateTime

from .base import Base

class BloodPressureReadings(Base):
    __tablename__ = "blood_pressure_readings"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=True)
    systolic = Column(Integer, nullable=True)
    diastolic = Column(Integer, nullable=True)
    hrv = Column(Integer, nullable=True)
    stress = Column(Integer, nullable=True)
    patient_id = Column(Integer, nullable=True)
    actual_time = Column(DateTime, nullable=True)
