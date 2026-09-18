#!/usr/bin/env python3
"""
Patient Doctor Mapping Service for Revival Medical System
Handles patient-doctor relationship database READ operations only
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from dal.models.patient_doctor_mapping import PatientDoctorMapping
from dal.models.users import Users

logger = logging.getLogger(__name__)

class PatientDoctorMappingService:
    """Service class for patient-doctor mapping related database READ operations only"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def get_patient_doctors(self, patient_id: int, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all doctors assigned to a specific patient
        
        Args:
            patient_id (int): Patient ID
            active_only (bool): If True, only return active mappings
            
        Returns:
            List[Dict]: List of doctor information
        """
        try:
            current_date = datetime.now()
            
            query = self.db.query(
                PatientDoctorMapping.user_id,
                PatientDoctorMapping.patient_id,
                PatientDoctorMapping.from_date,
                PatientDoctorMapping.to_date,
                PatientDoctorMapping.is_primary,
                Users.first_name.label('doctor_first_name'),
                Users.last_name.label('doctor_last_name'),
                Users.email.label('doctor_email'),
                Users.role_id.label('doctor_role_id')
            ).join(
                Users, PatientDoctorMapping.user_id == Users.id
            ).filter(
                PatientDoctorMapping.patient_id == patient_id
            )
            
            if active_only:
                query = query.filter(
                    and_(
                        or_(
                            PatientDoctorMapping.from_date.is_(None),
                            PatientDoctorMapping.from_date <= current_date
                        ),
                        or_(
                            PatientDoctorMapping.to_date.is_(None),
                            PatientDoctorMapping.to_date >= current_date
                        )
                    )
                )
            
            results = query.order_by(
                PatientDoctorMapping.is_primary.desc(),  # Primary doctors first
                PatientDoctorMapping.from_date.desc()
            ).all()
            
            doctors = []
            for result in results:
                full_name = f"{result.doctor_first_name or ''} {result.doctor_last_name or ''}".strip()
                doctors.append({
                    'user_id': result.user_id,
                    'patient_id': result.patient_id,
                    'doctor_name': full_name,
                    'doctor_first_name': result.doctor_first_name,
                    'doctor_last_name': result.doctor_last_name,
                    'doctor_email': result.doctor_email,
                    'doctor_role_id': result.doctor_role_id,
                    'from_date': result.from_date.isoformat() if result.from_date else None,
                    'to_date': result.to_date.isoformat() if result.to_date else None,
                    'is_primary': bool(result.is_primary),
                    'is_active': self._is_mapping_active(result.from_date, result.to_date)
                })
            
            logger.info(f"Retrieved {len(doctors)} doctors for patient {patient_id}")
            return doctors
            
        except Exception as e:
            logger.error(f"Error retrieving doctors for patient {patient_id}: {e}")
            return []
    
    def get_doctor_patients(self, doctor_user_id: int, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all patients assigned to a specific doctor
        
        Args:
            doctor_user_id (int): Doctor's user ID
            active_only (bool): If True, only return active mappings
            
        Returns:
            List[Dict]: List of patient information
        """
        try:
            current_date = datetime.now()
            
            query = self.db.query(
                PatientDoctorMapping.user_id,
                PatientDoctorMapping.patient_id,
                PatientDoctorMapping.from_date,
                PatientDoctorMapping.to_date,
                PatientDoctorMapping.is_primary,
                Users.first_name.label('patient_first_name'),
                Users.last_name.label('patient_last_name'),
                Users.email.label('patient_email')
            ).join(
                Users, PatientDoctorMapping.patient_id == Users.id
            ).filter(
                PatientDoctorMapping.user_id == doctor_user_id
            )
            
            if active_only:
                query = query.filter(
                    and_(
                        or_(
                            PatientDoctorMapping.from_date.is_(None),
                            PatientDoctorMapping.from_date <= current_date
                        ),
                        or_(
                            PatientDoctorMapping.to_date.is_(None),
                            PatientDoctorMapping.to_date >= current_date
                        )
                    )
                )
            
            results = query.order_by(
                PatientDoctorMapping.is_primary.desc(),  # Primary assignments first
                PatientDoctorMapping.from_date.desc()
            ).all()
            
            patients = []
            for result in results:
                full_name = f"{result.patient_first_name or ''} {result.patient_last_name or ''}".strip()
                patients.append({
                    'user_id': result.user_id,
                    'patient_id': result.patient_id,
                    'patient_name': full_name,
                    'patient_first_name': result.patient_first_name,
                    'patient_last_name': result.patient_last_name,
                    'patient_email': result.patient_email,
                    'from_date': result.from_date.isoformat() if result.from_date else None,
                    'to_date': result.to_date.isoformat() if result.to_date else None,
                    'is_primary': bool(result.is_primary),
                    'is_active': self._is_mapping_active(result.from_date, result.to_date)
                })
            
            logger.info(f"Retrieved {len(patients)} patients for doctor {doctor_user_id}")
            return patients
            
        except Exception as e:
            logger.error(f"Error retrieving patients for doctor {doctor_user_id}: {e}")
            return []
    
    def get_primary_doctor(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the primary doctor for a patient
        
        Args:
            patient_id (int): Patient ID
            
        Returns:
            Dict or None: Primary doctor information
        """
        try:
            current_date = datetime.now()
            
            result = self.db.query(
                PatientDoctorMapping.user_id,
                PatientDoctorMapping.patient_id,
                PatientDoctorMapping.from_date,
                PatientDoctorMapping.to_date,
                Users.first_name.label('doctor_first_name'),
                Users.last_name.label('doctor_last_name'),
                Users.email.label('doctor_email'),
                Users.role_id.label('doctor_role_id')
            ).join(
                Users, PatientDoctorMapping.user_id == Users.id
            ).filter(
                and_(
                    PatientDoctorMapping.patient_id == patient_id,
                    PatientDoctorMapping.is_primary == 1,
                    or_(
                        PatientDoctorMapping.from_date.is_(None),
                        PatientDoctorMapping.from_date <= current_date
                    ),
                    or_(
                        PatientDoctorMapping.to_date.is_(None),
                        PatientDoctorMapping.to_date >= current_date
                    )
                )
            ).first()
            
            if result:
                full_name = f"{result.doctor_first_name or ''} {result.doctor_last_name or ''}".strip()
                return {
                    'user_id': result.user_id,
                    'patient_id': result.patient_id,
                    'doctor_name': full_name,
                    'doctor_first_name': result.doctor_first_name,
                    'doctor_last_name': result.doctor_last_name,
                    'doctor_email': result.doctor_email,
                    'doctor_role_id': result.doctor_role_id,
                    'from_date': result.from_date.isoformat() if result.from_date else None,
                    'to_date': result.to_date.isoformat() if result.to_date else None,
                    'is_primary': True,
                    'is_active': True
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving primary doctor for patient {patient_id}: {e}")
            return None
    
    def check_doctor_patient_access(self, doctor_user_id: int, patient_id: int) -> bool:
        """
        Check if a doctor has access to a specific patient
        
        Args:
            doctor_user_id (int): Doctor's user ID
            patient_id (int): Patient ID
            
        Returns:
            bool: True if doctor has access, False otherwise
        """
        try:
            current_date = datetime.now()
            
            mapping = self.db.query(PatientDoctorMapping).filter(
                and_(
                    PatientDoctorMapping.user_id == doctor_user_id,
                    PatientDoctorMapping.patient_id == patient_id,
                    or_(
                        PatientDoctorMapping.from_date.is_(None),
                        PatientDoctorMapping.from_date <= current_date
                    ),
                    or_(
                        PatientDoctorMapping.to_date.is_(None),
                        PatientDoctorMapping.to_date >= current_date
                    )
                )
            ).first()
            
            return mapping is not None
            
        except Exception as e:
            logger.error(f"Error checking doctor-patient access for doctor {doctor_user_id}, patient {patient_id}: {e}")
            return False
    
    def _is_mapping_active(self, from_date: Optional[datetime], to_date: Optional[datetime]) -> bool:
        """Check if a mapping is currently active based on dates"""
        if not from_date:
            return True
        
        now = datetime.now()
        
        # If from_date is in the future, not active yet
        if from_date > now:
            return False
        
        # If to_date is in the past, no longer active
        if to_date and to_date < now:
            return False
        
        return True
