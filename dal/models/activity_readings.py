from sqlalchemy import Column, Integer, Float, String, Date

from .base import Base

class ActivityReadings(Base):
    __tablename__ = "activity_readings"
    id = Column(Integer, primary_key=True, index=True)
    total_exercise_duration = Column(Float, nullable=True)
    total_calories_burned = Column(Float, nullable=True)
    patient_id = Column(Integer, nullable=True)
    activity_type = Column(String(255), nullable=True)
    date = Column(Date, nullable=True)
    total_distance = Column(Float, nullable=True)
    total_step = Column(Integer, nullable=True)
