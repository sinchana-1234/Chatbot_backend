#!/usr/bin/env python3
"""
Device Service - Revival Medical System
Service layer for device management and expiry logic
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from dal.models.devices import Devices
from dal.models.users import Users
from dal.database import DatabaseManager


class DeviceService:
    """Service class for device-related operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def _resolve_patient_name_to_id(self, session: Session, patient_name: str, role: str, user_id: int) -> Optional[int]:
        """
        Resolve patient name to patient ID based on role permissions
        """
        if role == 'patient':
            # Patients can only see their own data
            user = session.query(Users).filter_by(id=user_id).first()
            if user and user.name.lower() == patient_name.lower():
                return user_id
            return None
        elif role in ['doctor', 'staff']:
            # Doctors and staff can see all patients
            user = session.query(Users).filter(Users.name.ilike(f'%{patient_name}%')).first()
            return user.id if user else None
        return None
    
    def get_device_by_id(self, device_id: int, role: str, user_id: int) -> Optional[Dict[str, Any]]:
        """Get a device by ID with role-based access control"""
        try:
            with self.db_manager as db_mgr:
                if not db_mgr.db:
                    return None
                    
                device = db_mgr.db.query(Devices).filter_by(id=device_id).first()
                
                if not device:
                    return None
                
                # Role-based access control
                if role == 'patient' and device.patient_id != user_id:
                    return None
                
                device_dict = device.to_dict()
                device_dict['is_expired'] = device.is_expired
                device_dict['expiry_date'] = device.expiry_date.isoformat() if device.expiry_date else None
                device_dict['days_until_expiry'] = device.days_until_expiry
                
                return device_dict
        except Exception as e:
            print(f"Error getting device by ID: {e}")
            return None
    
    def get_devices_for_patient(self, patient_id: int, role: str, user_id: int, device_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all devices for a patient with role-based access control"""
        try:
            with self.db_manager as db_mgr:
                if not db_mgr.db:
                    return []
                    
                # Role-based access control
                if role == 'patient' and patient_id != user_id:
                    return []
                
                query = db_mgr.db.query(Devices).filter_by(patient_id=patient_id)
                
                if device_name:
                    query = query.filter(Devices.name.ilike(f'%{device_name}%'))
                
                devices = query.all()
                results = []
                for device in devices:
                    device_dict = device.to_dict()
                    device_dict['is_expired'] = device.is_expired
                    device_dict['expiry_date'] = device.expiry_date.isoformat() if device.expiry_date else None
                    device_dict['days_until_expiry'] = device.days_until_expiry
                    results.append(device_dict)
                
                return results
        except Exception as e:
            print(f"Error getting devices for patient: {e}")
            return []
    
    def get_cgm_devices(self, patient_name: Optional[str], role: str, user_id: int) -> List[Dict[str, Any]]:
        """Get CGM devices with expiry information"""
        try:
            with self.db_manager as db_mgr:
                if not db_mgr.db:
                    return []
                    
                # Build query for CGM devices
                query = db_mgr.db.query(Devices).filter(
                    Devices.name.ilike('%cgm%'),
                    Devices.status == 1
                )
                
                # Apply role-based filtering
                if role == 'patient':
                    query = query.filter_by(patient_id=user_id)
                elif patient_name and role in ['doctor', 'staff']:
                    patient_id = self._resolve_patient_name_to_id(db_mgr.db, patient_name, role, user_id)
                    if patient_id:
                        query = query.filter_by(patient_id=patient_id)
                    else:
                        return []
                
                devices = query.all()
                
                # Enhance with expiry information
                results = []
                for device in devices:
                    device_dict = device.to_dict()
                    device_dict['is_expired'] = device.is_expired
                    device_dict['expiry_date'] = device.expiry_date.isoformat() if device.expiry_date else None
                    device_dict['days_until_expiry'] = device.days_until_expiry
                    
                    # Get patient name for display
                    patient = db_mgr.db.query(Users).filter_by(id=device.patient_id).first()
                    device_dict['patient_name'] = patient.name if patient else 'Unknown'
                    
                    results.append(device_dict)
                
                return results
        except Exception as e:
            print(f"Error getting CGM devices: {e}")
            return []
    
    def check_device_expiry(self, patient_name: Optional[str], device_name: str, role: str, user_id: int) -> Dict[str, Any]:
        """Check when a specific device expires"""
        try:
            with self.db_manager as db_mgr:
                if not db_mgr.db:
                    return {
                        'success': False,
                        'message': 'Database connection not available'
                    }
                    
                # Build base query
                query = db_mgr.db.query(Devices).filter(
                    Devices.name.ilike(f'%{device_name}%'),
                    Devices.status == 1
                )
                
                # Apply role-based filtering
                if role == 'patient':
                    query = query.filter_by(patient_id=user_id)
                    patient_user = db_mgr.db.query(Users).filter_by(id=user_id).first()
                    display_patient_name = patient_user.name if patient_user else 'You'
                elif patient_name and role in ['doctor', 'staff']:
                    patient_id = self._resolve_patient_name_to_id(db_mgr.db, patient_name, role, user_id)
                    if patient_id:
                        query = query.filter_by(patient_id=patient_id)
                        display_patient_name = patient_name
                    else:
                        return {
                            'success': False,
                            'message': f"Patient '{patient_name}' not found or access denied."
                        }
                else:
                    return {
                        'success': False,
                        'message': "Patient name is required for doctors and staff."
                    }
                
                device = query.first()
                
                if not device:
                    return {
                        'success': False,
                        'message': f"No active {device_name} device found for {display_patient_name}."
                    }
                
                # Get expiry information
                expiry_date = device.expiry_date
                days_until_expiry = device.days_until_expiry
                is_expired = device.is_expired
                
                if is_expired:
                    if expiry_date:
                        return {
                            'success': True,
                            'message': f"{display_patient_name}'s {device_name} expired on {expiry_date.strftime('%Y-%m-%d')}.",
                            'device_name': device_name,
                            'patient_name': display_patient_name,
                            'expiry_date': expiry_date.isoformat(),
                            'is_expired': True,
                            'days_until_expiry': 0
                        }
                    else:
                        return {
                            'success': True,
                            'message': f"{display_patient_name}'s {device_name} has expired (no session start date found).",
                            'device_name': device_name,
                            'patient_name': display_patient_name,
                            'is_expired': True
                        }
                else:
                    if expiry_date:
                        if days_until_expiry is not None:
                            if days_until_expiry <= 3:
                                urgency = "expires very soon"
                            elif days_until_expiry <= 7:
                                urgency = "expires soon"
                            else:
                                urgency = "expires"
                            
                            return {
                                'success': True,
                                'message': f"{display_patient_name}'s {device_name} {urgency} on {expiry_date.strftime('%Y-%m-%d')} ({days_until_expiry} days remaining).",
                                'device_name': device_name,
                                'patient_name': display_patient_name,
                                'expiry_date': expiry_date.isoformat(),
                                'is_expired': False,
                                'days_until_expiry': days_until_expiry
                            }
                        else:
                            return {
                                'success': True,
                                'message': f"{display_patient_name}'s {device_name} expires on {expiry_date.strftime('%Y-%m-%d')}.",
                                'device_name': device_name,
                                'patient_name': display_patient_name,
                                'expiry_date': expiry_date.isoformat(),
                                'is_expired': False
                            }
                    else:
                        return {
                            'success': True,
                            'message': f"{display_patient_name}'s {device_name} is active but no expiry date is set.",
                            'device_name': device_name,
                            'patient_name': display_patient_name,
                            'is_expired': False
                        }
                
        except Exception as e:
            print(f"Error checking device expiry: {e}")
            return {
                'success': False,
                'message': f"Error checking device expiry: {str(e)}"
            }
    
    def get_all_devices_for_user(self, role: str, user_id: int) -> List[Dict[str, Any]]:
        """Get all devices visible to the user based on their role"""
        try:
            with self.db_manager as db_mgr:
                if not db_mgr.db:
                    return []
                    
                if role == 'patient':
                    # Patients see only their own devices
                    devices = db_mgr.db.query(Devices).filter_by(patient_id=user_id).all()
                else:
                    # Doctors and staff see all devices
                    devices = db_mgr.db.query(Devices).all()
                
                results = []
                for device in devices:
                    device_dict = device.to_dict()
                    device_dict['is_expired'] = device.is_expired
                    device_dict['expiry_date'] = device.expiry_date.isoformat() if device.expiry_date else None
                    device_dict['days_until_expiry'] = device.days_until_expiry
                    
                    # Get patient name for display
                    patient = db_mgr.db.query(Users).filter_by(id=device.patient_id).first()
                    device_dict['patient_name'] = patient.name if patient else 'Unknown'
                    
                    results.append(device_dict)
                
                return results
        except Exception as e:
            print(f"Error getting all devices: {e}")
            return []


# Create a global instance
device_service = None

def get_device_service() -> DeviceService:
    """Get the global device service instance"""
    global device_service
    if device_service is None:
        from dal.database import DatabaseManager
        db_manager = DatabaseManager()
        device_service = DeviceService(db_manager)
    return device_service
