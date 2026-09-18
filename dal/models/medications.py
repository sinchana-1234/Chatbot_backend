from sqlalchemy import Column, Integer, String, DateTime, Text

from .base import Base

class Medications(Base):
    __tablename__ = "medications"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, nullable=True)
    medication_type = Column(String(255), nullable=True)
    medication_name = Column(String(500), nullable=True)
    dosage = Column(String(255), nullable=True)
    frequency = Column(String(255), nullable=True)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    note = Column(Text, nullable=True)
    created = Column(DateTime, nullable=True)
    created_by = Column(Integer, nullable=True)
    progress = Column(String(50), nullable=True)
    status = Column(Integer, nullable=True, default=1)
