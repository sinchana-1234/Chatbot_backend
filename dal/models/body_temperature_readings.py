from sqlalchemy import Column, Integer, Float, DateTime

from .base import Base

class BodyTemperatureReadings(Base):
    __tablename__ = "body_temperature_readings"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=True)
    temperature = Column(Float, nullable=True)
    patient_id = Column(Integer, nullable=True)
    actual_time = Column(DateTime, nullable=True)
