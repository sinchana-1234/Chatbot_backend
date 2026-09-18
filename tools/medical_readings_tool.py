#!/usr/bin/env python3
"""
Medical Readings Tool
Basic tool for getting comprehensive medical readings for a patient
"""

import logging
import json
from typing import Optional
from datetime import datetime
from langchain.tools import BaseTool

# Import our medical system components
from dal.database import DatabaseManager

logger = logging.getLogger(__name__)

class MedicalReadingsTool(BaseTool):
    """Tool for getting comprehensive medical readings for a patient"""
    name: str = "get_medical_readings"
    description: str = """Get comprehensive medical readings for a patient including glucose, blood pressure, temperature, activity, HRV, SpO2, stress readings, and current medications/supplements. 
    
    Parameters:
    - patient_id (int): Patient ID number (required if patient_name not provided)
    - patient_name (str): Patient name (required if patient_id not provided) 
    - start_date (str): Start date in YYYY-MM-DD format (optional)
    - end_date (str): End date in YYYY-MM-DD format (optional)
    
    If no dates provided, returns the latest readings. This tool provides a comprehensive overview including:
    - All vital signs and health metrics
    - Current medications and supplements
    - Recent activity data
    
    Use this tool to answer any questions about patient medical data and medication status.

    IMPORTANT: DO NOT use this tool for protocol, treatment plan, or guideline queries. For any protocol/treatment/guideline-related questions, ALWAYS use the get_protocols tool instead. This tool does NOT return protocol/treatment/guideline information."""
    
    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None, 
             start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        """Get medical readings for a patient"""
        try:
            # Parse dates if provided
            start_datetime = None
            end_datetime = None
            
            if start_date:
                try:
                    start_datetime = datetime.fromisoformat(start_date)
                except ValueError:
                    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            
            if end_date:
                try:
                    end_datetime = datetime.fromisoformat(end_date)
                except ValueError:
                    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Get medical readings using context manager
            with DatabaseManager() as db_manager:
                result = db_manager.get_medical_readings(
                    patient_id=patient_id,
                    patient_name=patient_name,
                    start_date=start_datetime,
                    end_date=end_datetime
                )
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error in MedicalReadingsTool: {e}")
            return f"Error getting medical readings: {str(e)}"
