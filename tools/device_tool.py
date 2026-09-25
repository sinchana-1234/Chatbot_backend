#!/usr/bin/env python3
"""
Device Tool for Revival Medical System
Simple tool for checking device expiry and counting devices per patient
"""

import logging
import json
from typing import Optional, Union
from datetime import datetime, timedelta
from langchain.tools import BaseTool
from dal.database import DatabaseManager
from dal.models.devices import Devices
from dal.models.users import Users

logger = logging.getLogger(__name__)

class DeviceTool(BaseTool):
    """Tool for checking device expiry status and counting devices"""
    name: str = "check_device_status"
    description: str = """Check device expiry status and count devices for a patient.
    
    Parameters:
    - patient_identifier (str): Patient name or patient ID (required)
    - device_name (str): Specific device name to check (optional, defaults to "CGM")
    - check_all_devices (bool): Set to true to get all devices for the patient (optional)
    
    Use this tool for queries like:
    - "When does my CGM expire?" 
    - "Is my CGM expired?"
    - "How many devices does patient 132 have?"
    - "Show all devices for Rayudu"
    - "Check CGM status for patient 111"
    
    Returns:
    - Device expiry status (expired or not)
    - Number of devices per patient
    - Device details with expiry information
    """
    
    def __init__(self):
        super().__init__()
    
    def set_user_context(self, user_context):
        """Set user context for role-based access control"""
        object.__setattr__(self, 'user_context', user_context)
    
    def _get_patient_id(self, patient_identifier: Union[str, int], db_session) -> Optional[int]:
        """Convert patient name or ID to patient ID"""
        try:
            # If it's already a number, return it
            if isinstance(patient_identifier, int):
                return patient_identifier
            
            # Try to convert string to int (patient ID)
            try:
                return int(patient_identifier)
            except ValueError:
                pass
            
            # Search by patient name (first_name or last_name)
            patient = db_session.query(Users).filter(
                (Users.first_name.ilike(f'%{patient_identifier}%')) |
                (Users.last_name.ilike(f'%{patient_identifier}%'))
            ).first()
            
            return patient.id if patient else None
            
        except Exception as e:
            logger.error(f"Error resolving patient identifier: {e}")
            return None
    
    def _run(self, patient_identifier: str, device_name: str = "CGM", 
             check_all_devices: bool = False) -> str:
        """Check device status with simplified logic"""
        try:
            # Get user context for role-based access
            user_context = getattr(self, 'user_context', None)
            if not user_context:
                return json.dumps({
                    "success": False,
                    "error": "User context not available"
                })
            
            role_id = user_context.get('role_id')
            current_user_id = user_context.get('user_id')
            
            with DatabaseManager() as db_manager:
                if not db_manager.db:
                    return json.dumps({
                        "success": False,
                        "error": "Database connection not available"
                    })
                
                # Role-based access control. A patient may ONLY see their own devices,
                # so FORCE the id to their own user_id and ignore whatever patient was
                # named — the same pattern every other tool uses. (The old check
                # compared user_context.get('role','') to 'patient', but the context
                # carries 'role_id'/'role_name', not 'role', so the guard never fired
                # and a patient could read another patient's device.)
                if role_id == 1:  # Patient — locked to their own data
                    patient_id = current_user_id
                else:             # Staff — may look up any patient by name/ID
                    patient_id = self._get_patient_id(patient_identifier, db_manager.db)
                
                if not patient_id:
                    return json.dumps({
                        "success": False,
                        "message": f"Patient '{patient_identifier}' not found"
                    })
                
                # Get patient name for display
                patient = db_manager.db.query(Users).filter_by(id=patient_id).first()
                if patient:
                    patient_name = f"{patient.first_name or ''} {patient.last_name or ''}".strip()
                    if not patient_name:
                        patient_name = f"Patient {patient_id}"
                else:
                    patient_name = f"Patient {patient_id}"
                
                if check_all_devices:
                    # Get only active devices for the patient
                    devices = db_manager.db.query(Devices).filter(
                        Devices.patient_id == patient_id,
                        Devices.status == 1  # Only active devices
                    ).all()
                    
                    device_list = []
                    expired_count = 0
                    
                    for device in devices:
                        device_info = {
                            "id": device.id,
                            "name": device.name,
                            "tag_id": device.tag_id,
                            "status": "Active",  # All devices are active now
                            "mapped_date": device.mapped_date.isoformat() if device.mapped_date else None,
                            "session_start_date": device.session_start_date.isoformat() if device.session_start_date else None,
                            "is_expired": device.is_expired,
                            "expiry_date": device.expiry_date.isoformat() if device.expiry_date else None,
                            "days_until_expiry": device.days_until_expiry
                        }
                        
                        device_list.append(device_info)
                        
                        if device.is_expired:
                            expired_count += 1
                    
                    return json.dumps({
                        "success": True,
                        "patient_name": patient_name,
                        "patient_id": patient_id,
                        "total_active_devices": len(devices),
                        "expired_devices": expired_count,
                        "devices": device_list
                    }, indent=2)
                
                else:
                    # Check a specific device (e.g. CGM). For "is my CGM expired?" we
                    # must look at the patient's MOST RECENT matching device REGARDLESS
                    # of status — an expired CGM is status != 1, so the old
                    # "status == 1" filter hid it and wrongly reported "no device
                    # found" for a patient who does have one. Order by
                    # session_start_date so the newest session is the one we report on.
                    device = db_manager.db.query(Devices).filter(
                        Devices.patient_id == patient_id,
                        Devices.name.ilike(f'%{device_name}%')
                    ).order_by(Devices.session_start_date.desc()).first()
                    
                    if not device:
                        # Genuinely no such device ever — not merely "none active".
                        return json.dumps({
                            "success": False,
                            "message": f"No {device_name} device found for {patient_name}"
                        })
                    
                    # Calculate expiry information (computed from session_start_date + 15 days)
                    is_expired = device.is_expired
                    expiry_date = device.expiry_date
                    days_until_expiry = device.days_until_expiry
                    is_active = (device.status == 1)
                    
                    result = {
                        "success": True,
                        "patient_name": patient_name,
                        "patient_id": patient_id,
                        "device_name": device.name,
                        "device_id": device.id,
                        "tag_id": device.tag_id,
                        "is_active": is_active,
                        "is_expired": is_expired,
                        "expiry_date": expiry_date.isoformat() if expiry_date else None,
                        "days_until_expiry": days_until_expiry,
                        "session_start_date": device.session_start_date.isoformat() if device.session_start_date else None,
                        "mapped_date": device.mapped_date.isoformat() if device.mapped_date else None
                    }
                    
                    # Add friendly message
                    if is_expired:
                        if expiry_date:
                            result["message"] = f"{patient_name}'s most recent {device.name} expired on {expiry_date.strftime('%d-%m-%Y')}."
                        else:
                            result["message"] = f"{patient_name}'s most recent {device.name} has expired."
                    elif days_until_expiry is not None:
                        result["message"] = f"{patient_name}'s {device.name} is active and expires in {days_until_expiry} days ({expiry_date.strftime('%d-%m-%Y')})."
                    else:
                        result["message"] = f"{patient_name}'s {device.name} is active but no expiry date is available."
                    
                    return json.dumps(result, indent=2)
                    
        except Exception as e:
            logger.error(f"Error in DeviceTool._run: {e}")
            return json.dumps({
                "success": False,
                "error": f"Error checking device status: {str(e)}"
            })
    
    async def _arun(self, patient_identifier: str, device_name: str = "CGM", 
                    check_all_devices: bool = False) -> str:
        """Async version of the tool"""
        return self._run(patient_identifier, device_name, check_all_devices)
