#!/usr/bin/env python3
"""
Plan Service for Revival Medical System
Handles plan master and my plan related database operations
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from dal.models.plan_master import PlanMaster
from dal.models.my_plan import MyPlan
from dal.models.users import Users
from .base_service import BaseService

logger = logging.getLogger(__name__)

class PlanService(BaseService):
    """Service class for plan-related database operations"""
    
    def __init__(self, db_session: Session):
        super().__init__(db_session)
    
    def get_user_plans(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all plans for a specific patient with plan details
        
        Args:
            patient_id (int): Patient ID (optional if patient_name provided)
            patient_name (str): Patient name (optional if patient_id provided)
            active_only (bool): Whether to return only active plans
            
        Returns:
            List[Dict]: List of plan details with master plan information
        """
        try:
            logger.warning(f"Patient name: patient_name={patient_name}")
                
            # Find patient ID
            patient_id = self.find_patient_by_name_or_id(patient_id, patient_name)
            if not patient_id:
                logger.warning(f"Patient not found: patient_id={patient_id}, patient_name={patient_name}")
                return []
                
            # Build query with join
            query = self.db.query(
                MyPlan.id.label('my_plan_id'),
                MyPlan.purched_date,
                MyPlan.from_date,
                MyPlan.to_date,
                MyPlan.status.label('plan_status'),
                MyPlan.available_doctor_consultation,
                MyPlan.available_hc_consultation,
                MyPlan.consumed_doctor_consultation,
                MyPlan.consumed_hc_consultation,
                PlanMaster.id.label('plan_id'),
                PlanMaster.name.label('plan_name'),
                PlanMaster.price,
                PlanMaster.plan_duration,
                PlanMaster.description,
                PlanMaster.no_of_doctor_consultant,
                PlanMaster.no_of_health_controller,
                PlanMaster.plan_type,
                PlanMaster.product_name,
                PlanMaster.cgm_unit,
                PlanMaster.bio_sensor_unit
            ).join(
                PlanMaster, MyPlan.plan_id == PlanMaster.id
            ).filter(
                MyPlan.patient_id == patient_id
            )
            
            # Filter for active plans only if requested
            if active_only:
                current_date = datetime.now()
                query = query.filter(
                    and_(
                        MyPlan.status == 1,  # Active status
                        or_(
                            MyPlan.to_date >= current_date,
                            MyPlan.to_date.is_(None)
                        )
                    )
                )
            
            # Order by purchase date (most recent first)
            query = query.order_by(MyPlan.purched_date.desc())
            
            results = query.all()
            
            plans = []
            for row in results:
                plan_data = {
                    'my_plan_id': row.my_plan_id,
                    'plan_id': row.plan_id,
                    'plan_name': row.plan_name,
                    'plan_type': row.plan_type,
                    'price': row.price,
                    'plan_duration': row.plan_duration,
                    'description': row.description,
                    'product_name': row.product_name,
                    'purchase_date': row.purched_date.isoformat() if row.purched_date else None,
                    'from_date': row.from_date.isoformat() if row.from_date else None,
                    'to_date': row.to_date.isoformat() if row.to_date else None,
                    'plan_status': 'Active' if row.plan_status == 1 else 'Inactive',
                    'total_doctor_consultations': row.no_of_doctor_consultant,
                    'total_hc_consultations': row.no_of_health_controller,
                    'available_doctor_consultations': row.available_doctor_consultation,
                    'available_hc_consultations': row.available_hc_consultation,
                    'consumed_doctor_consultations': row.consumed_doctor_consultation,
                    'consumed_hc_consultations': row.consumed_hc_consultation,
                    'cgm_units': row.cgm_unit,
                    'bio_sensor_units': row.bio_sensor_unit,
                    'is_current': self._is_current_plan(row.from_date, row.to_date)
                }
                plans.append(plan_data)
            
            logger.info(f"Retrieved {len(plans)} plans for patient {patient_id}")
            return plans
            
        except Exception as e:
            logger.error(f"Error retrieving plans for patient {patient_id}: {e}")
            return []
    
    def get_current_active_plan(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the current active plan for a patient
        
        Args:
            patient_id (int): Patient ID (optional if patient_name provided)
            patient_name (str): Patient name (optional if patient_id provided)
            
        Returns:
            Dict or None: Current active plan details
        """
        try:
            # Find patient ID
            patient_id = self.find_patient_by_name_or_id(patient_id, patient_name)
            if not patient_id:
                logger.warning(f"Patient not found: patient_id={patient_id}, patient_name={patient_name}")
                return None
                
            logger.info(f"Getting current active plan for patient ID: {patient_id}")
            current_date = datetime.now()
            
            result = self.db.query(
                MyPlan.id.label('my_plan_id'),
                MyPlan.purched_date,
                MyPlan.from_date,
                MyPlan.to_date,
                MyPlan.status.label('plan_status'),
                MyPlan.available_doctor_consultation,
                MyPlan.available_hc_consultation,
                MyPlan.consumed_doctor_consultation,
                MyPlan.consumed_hc_consultation,
                PlanMaster.id.label('plan_id'),
                PlanMaster.name.label('plan_name'),
                PlanMaster.price,
                PlanMaster.plan_duration,
                PlanMaster.description,
                PlanMaster.no_of_doctor_consultant,
                PlanMaster.no_of_health_controller,
                PlanMaster.plan_type,
                PlanMaster.product_name
            ).join(
                PlanMaster, MyPlan.plan_id == PlanMaster.id
            ).filter(
                and_(
                    MyPlan.patient_id == patient_id,
                    MyPlan.status == 1,  # Active status
                    MyPlan.from_date <= current_date,
                    or_(
                        MyPlan.to_date >= current_date,
                        MyPlan.to_date.is_(None)
                    )
                )
            ).order_by(MyPlan.from_date.desc()).first()
            
            if result:
                return {
                    'my_plan_id': result.my_plan_id,
                    'plan_id': result.plan_id,
                    'plan_name': result.plan_name,
                    'plan_type': result.plan_type,
                    'price': result.price,
                    'plan_duration': result.plan_duration,
                    'description': result.description,
                    'product_name': result.product_name,
                    'purchase_date': result.purched_date.isoformat() if result.purched_date else None,
                    'from_date': result.from_date.isoformat() if result.from_date else None,
                    'to_date': result.to_date.isoformat() if result.to_date else None,
                    'plan_status': 'Active',
                    'total_doctor_consultations': result.no_of_doctor_consultant,
                    'total_hc_consultations': result.no_of_health_controller,
                    'available_doctor_consultations': result.available_doctor_consultation,
                    'available_hc_consultations': result.available_hc_consultation,
                    'consumed_doctor_consultations': result.consumed_doctor_consultation,
                    'consumed_hc_consultations': result.consumed_hc_consultation
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving current plan for patient {patient_id}: {e}")
            return None
    
    def get_plan_usage_summary(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get plan usage summary for a patient
        
        Args:
            patient_id (int): Patient ID (optional if patient_name provided)
            patient_name (str): Patient name (optional if patient_id provided)
            
        Returns:
            Dict: Plan usage summary
        """
        try:
            # Find patient ID
            patient_id = self.find_patient_by_name_or_id(patient_id, patient_name)
            if not patient_id:
                return {
                    'has_active_plan': False,
                    'message': 'Patient not found'
                }
                
            current_plan = self.get_current_active_plan(patient_id=patient_id)
            
            if not current_plan:
                return {
                    'has_active_plan': False,
                    'message': 'No active plan found'
                }
            
            # Calculate actual remaining consultations
            total_doctor = current_plan['total_doctor_consultations'] or 0
            total_hc = current_plan['total_hc_consultations'] or 0
            consumed_doctor = current_plan['consumed_doctor_consultations'] or 0
            consumed_hc = current_plan['consumed_hc_consultations'] or 0
            
            # Calculate remaining consultations properly
            remaining_doctor = max(0, total_doctor - consumed_doctor)
            remaining_hc = max(0, total_hc - consumed_hc)
            
            return {
                'has_active_plan': True,
                'plan_name': current_plan['plan_name'],
                'plan_type': current_plan['plan_type'],
                'validity': {
                    'from_date': current_plan['from_date'],
                    'to_date': current_plan['to_date']
                },
                'consultations': {
                    'doctor': {
                        'total': total_doctor,
                        'available': remaining_doctor,
                        'consumed': consumed_doctor,
                        'percentage_used': round((consumed_doctor / total_doctor) * 100, 2) if total_doctor > 0 else 0
                    },
                    'health_coach': {
                        'total': total_hc,
                        'available': remaining_hc,
                        'consumed': consumed_hc,
                        'percentage_used': round((consumed_hc / total_hc) * 100, 2) if total_hc > 0 else 0
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting plan usage summary for patient {patient_id}: {e}")
            return {
                'has_active_plan': False,
                'error': str(e)
            }
    
    def _is_current_plan(self, from_date: datetime, to_date: datetime) -> bool:
        """Check if a plan is currently active based on dates"""
        if not from_date:
            return False
        
        current_date = datetime.now()
        
        if to_date:
            return from_date <= current_date <= to_date
        else:
            return from_date <= current_date
