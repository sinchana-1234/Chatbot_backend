#!/usr/bin/env python3
"""
Plan Master Model - Revival Medical System
"""

from sqlalchemy import Column, Integer, String, DateTime, Numeric, Text, SmallInteger
from .base import Base

class PlanMaster(Base):
    """Plan Master table model"""
    __tablename__ = 'plan_master'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=True)
    price = Column(Numeric(10, 2), nullable=True)
    plan_duration = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    status = Column(SmallInteger, nullable=False, default=0)
    item_id = Column(String(255), nullable=True)
    item_id_old = Column(String(255), nullable=True)
    created = Column(DateTime, nullable=True)
    updated = Column(DateTime, nullable=True)
    no_of_doctor_consultant = Column(Integer, nullable=True, default=0)
    no_of_health_controller = Column(Integer, nullable=True, default=0)
    plan_type = Column(String(50), nullable=True, default='plan')
    report_code = Column(String(100), nullable=True)
    project_code = Column(String(100), nullable=True)
    product_name = Column(String(500), nullable=True)
    consumption_type = Column(SmallInteger, nullable=False, default=0)
    tax_range = Column(Numeric(6, 2), nullable=True)
    discount = Column(Numeric(6, 2), nullable=True)
    hsn_code = Column(String(30), nullable=True)
    product_rate = Column(Integer, nullable=True, default=1)
    cgm_unit = Column(Integer, nullable=True)
    bio_sensor_unit = Column(Integer, nullable=True)
    stock_in_hand = Column(Integer, nullable=True, default=999999)
    reorder_point = Column(Integer, nullable=True, default=50)
    addon_type = Column(String(255), nullable=True)

    def __repr__(self):
        return f"<PlanMaster(id={self.id}, name='{self.name}', price={self.price})>"
