from sqlalchemy import Column, Integer, String, Date, DateTime

from .base import Base

class Users(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    mobile_number = Column(String(255), nullable=True)
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    state = Column(String(255), nullable=True)
    zipcode = Column(String(255), nullable=True)
    role_id = Column(Integer, nullable=False)
    dob = Column(Date, nullable=True)
    sex = Column(String(255), nullable=True)
    created = Column(DateTime, nullable=True)
    updated = Column(DateTime, nullable=True)
    password = Column(String(255), nullable=True)
    status = Column(Integer, nullable=False)
    address = Column(String(255), nullable=True)
    city = Column(String(255), nullable=True)
    profile = Column(String(255), nullable=True)
    customer_id = Column(String(255), nullable=True)
    customer_id_old = Column(String(255), nullable=True)
    contact_person_id = Column(String(255), nullable=True)
    contact_person_id_old = Column(String(255), nullable=True)
    scheduled_status = Column(Integer, nullable=True, default=0)
    token = Column(String(255), nullable=True)
