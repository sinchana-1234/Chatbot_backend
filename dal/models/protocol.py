from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Protocol(Base):
    __tablename__ = 'protocol'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    doctor_id = Column(Integer, nullable=False)
    patient_id = Column(Integer, nullable=False)
    createdon = Column(DateTime, nullable=True)
    createdby = Column(Integer, nullable=True)
    status = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)  # Contains HTML formatted protocol content
    
    def __repr__(self):
        return f"<Protocol(id={self.id}, patient_id={self.patient_id}, doctor_id={self.doctor_id})>"
