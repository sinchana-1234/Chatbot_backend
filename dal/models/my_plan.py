#!/usr/bin/env python3
"""
My Plan Model - Revival Medical System
"""

from sqlalchemy import Column, Integer, DateTime, SmallInteger
from .base import Base

class MyPlan(Base):
    """My Plan table model"""
    __tablename__ = 'my_plan'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    purched_date = Column(DateTime, nullable=True)
    from_date = Column(DateTime, nullable=True)
    to_date = Column(DateTime, nullable=True)
    status = Column(SmallInteger, nullable=False, default=0)
    plan_id = Column(Integer, nullable=False)
    patient_id = Column(Integer, nullable=False)
    available_doctor_consultation = Column(Integer, nullable=True, default=0)
    available_hc_consultation = Column(Integer, nullable=True, default=0)
    consumed_doctor_consultation = Column(Integer, nullable=True, default=0)
    consumed_hc_consultation = Column(Integer, nullable=True, default=0)

    def __repr__(self):
        return f"<MyPlan(id={self.id}, patient_id={self.patient_id}, plan_id={self.plan_id})>"
