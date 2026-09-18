

from sqlalchemy import Column, Integer, Float, DateTime, String

from .base import Base

class SleepReadingsDetails(Base):
    __tablename__ = "sleep_readings_details"
    id = Column(Integer, primary_key=True, index=True)
    sleep_type = Column(String, nullable=True)
    date = Column(DateTime, nullable=True)
    value = Column(Float, nullable=True)
    level = Column(Integer, nullable=True)
    patient_id = Column(Integer, nullable=True)
