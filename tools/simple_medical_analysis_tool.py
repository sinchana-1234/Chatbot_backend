#!/usr/bin/env python3
"""
Simple Medical Analysis Tool
Simplified tool for basic medical analysis that works with existing database
"""

import logging
import json
from typing import Optional
from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

class SimpleMedicalAnalysisTool(BaseTool):
    """Simplified tool for basic medical analysis that works with existing database"""
    name: str = "get_basic_medical_analysis"
    description: str = """Get basic medical analysis using existing database functionality.
    
    Parameters:
    - patient_id (int): Patient ID number (optional if patient_name provided)
    - patient_name (str): Patient name (optional if patient_id provided) 
    - analysis_request (str): What analysis to perform - "medications", "food", "protocols", "sleep"
    
    Note: This returns a message that the requested functionality is not yet available.
    """
    
    def __init__(self):
        super().__init__()
        # Don't set user_context as instance variable to avoid Pydantic validation issues
    
    def set_user_context(self, user_context):
        """Set user context for role-based access control"""
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, 'user_context', user_context)
    
    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
             analysis_request: str = "medications") -> str:
        """Get basic medical analysis with role-based access control"""
        try:
            # Enforce role-based access control
            user_context = getattr(self, "user_context", None)
            if user_context and user_context.get('role_id') == 1:  # Patient role
                # Patients can only access their own analysis
                patient_id = user_context.get('user_id')
                patient_name = None  # Override any patient_name to enforce access control
            elif patient_id is None and patient_name is None:
                # For medical staff, if no patient specified, this might be an error
                return json.dumps({
                    "error": "Please specify a patient ID or patient name for the medical analysis."
                }, indent=2)
            
            return json.dumps({
                "message": f"The {analysis_request} analysis feature is not yet implemented in the database.",
                "available_features": [
                    "glucose readings",
                    "blood_pressure readings", 
                    "body_temperature readings",
                    "hrv readings",
                    "spo2 readings",
                    "stress readings",
                    "activity readings"
                ],
                "suggestion": f"Try asking for '{analysis_request}' readings using the general medical readings tool instead.",
                "patient_access": f"Query restricted to patient ID: {patient_id}" if user_context and user_context.get('role_id') == 1 else "Full access"
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error in SimpleMedicalAnalysisTool: {e}")
            return f"Error in basic medical analysis: {str(e)}"
