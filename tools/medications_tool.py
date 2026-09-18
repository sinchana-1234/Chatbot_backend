#!/usr/bin/env python3
"""
Medications Tool for Revival Medical System
"""

import logging
import json
from typing import Optional
from datetime import datetime
from langchain.tools import BaseTool
from dal.database import DatabaseManager

logger = logging.getLogger(__name__)

class MedicationsTool(BaseTool):
    """Tool for getting patient medications and supplements"""
    name: str = "get_medications"
    description: str = """Get current medications and supplements for a patient.
    
    Parameters:
    - patient_id (int): Patient ID number (optional if patient_name provided)
    - patient_name (str): Patient name (optional if patient_id provided)
    - medication_type (str): Filter by type - "medication", "supplement", or leave empty for all (optional)
    - date_filter (str): Date filter in YYYY-MM-DD format to get medications from that date onwards (optional)
    - limit (int): Maximum number of records to return (default: 10)
    
    Use this tool for queries like:
    - "List current medications for [patient]"
    - "List current supplements for [patient]" 
    - "What medications is [patient] taking?"
    - "What supplements is [patient] taking?"
    - "List latest medications for [patient]"
    
    Returns current active medications and/or supplements for the patient (top 10 latest by default).
    """
    
    def __init__(self):
        super().__init__()
        # Don't set user_context as instance variable to avoid Pydantic validation issues
    
    def set_user_context(self, user_context):
        """Set user context for role-based access control"""
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, 'user_context', user_context)
    
    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
             medication_type: Optional[str] = None, date_filter: Optional[str] = None,
             limit: int = 10) -> str:
        """Get patient medications with role-based access control"""
        try:
            # Enforce role-based access control
            user_context = getattr(self, 'user_context', None)
            if user_context and user_context.get('role_id') == 1:  # Patient role
                # Patients can only access their own medications
                patient_id = user_context.get('user_id')
                patient_name = None  # Override any patient_name to enforce access control
                logger.info(f"Patient access: restricting medications query to patient ID {patient_id}")
            elif patient_id is None and patient_name is None:
                # For medical staff, if no patient specified, this might be an error
                return "Please specify a patient ID or patient name for the medications query."
            
            # Parse date filter if provided
            parsed_date = None
            if date_filter:
                try:
                    parsed_date = datetime.strptime(date_filter, "%Y-%m-%d")
                except ValueError:
                    return json.dumps({
                        "error": f"Invalid date format. Use YYYY-MM-DD format. Got: {date_filter}"
                    }, indent=2)
            
            with DatabaseManager() as db_manager:
                result = db_manager.get_medications(
                    patient_id=patient_id,
                    patient_name=patient_name,
                    date_filter=parsed_date,
                    limit=limit
                )
                
                if "error" in result:
                    return json.dumps(result, indent=2)
                
                # Filter by medication type if specified
                if medication_type:
                    filtered_medications = []
                    filter_type = medication_type.lower().strip()
                    
                    # Handle different variations of the filter
                    if filter_type in ["supplement", "supplements"]:
                        filter_type = "supplement"
                    elif filter_type in ["medication", "medications", "medicine", "drug"]:
                        filter_type = "medication"
                    
                    for med in result.get("medications", []):
                        med_type = med.get("medication_type", "").lower().strip()
                        if med_type == filter_type:
                            filtered_medications.append(med)
                    
                    result["medications"] = filtered_medications
                    result["count"] = len(filtered_medications)
                    result["filter_applied"] = f"medication_type = {filter_type}"
                    result["message"] = f"Showing top {len(filtered_medications)} latest {filter_type}s" + (f" from {date_filter}" if date_filter else "")
                    
                    if len(filtered_medications) == 0:
                        result["message"] = f"No {filter_type}s found for this patient"
                
                return json.dumps(result, indent=2)
                
        except Exception as e:
            logger.error(f"Error in MedicationsTool: {e}")
            return json.dumps({
                "error": f"Error retrieving medications: {str(e)}"
            }, indent=2)
