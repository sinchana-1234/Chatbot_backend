from sqlalchemy import Column, Integer, String, DateTime, Text
from .base import Base

class Foodlog(Base):
    __tablename__ = "foodlog"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, nullable=False)
    type = Column(String(50), nullable=True)
    url = Column(String(500), nullable=True)
    activitydate = Column(String(50), nullable=True)
    createdon = Column(DateTime, nullable=True)
    createdby = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    status = Column(Integer, nullable=True, default=1)
    latitude = Column(String(100), nullable=True)
    longitude = Column(String(100), nullable=True)
