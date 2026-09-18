#!/usr/bin/env python3
"""
Multi-Patient Analysis Tool
Tool for analyzing readings across multiple patients with optional dates
"""

import logging
import json
from typing import Optional
from datetime import datetime
from langchain.tools import BaseTool

# Import our medical system components
from dal.database import DatabaseManager

logger = logging.getLogger(__name__)

class MultiPatientAnalysisTool(BaseTool):
    """Enhanced tool for analyzing readings across multiple patients with optional dates"""
    name: str = "analyze_multiple_patients"
    description: str = """Analyze medical readings across multiple patients to find distinct patients with high/low values.
    
    Parameters:
    - reading_type (str): Type of reading - "glucose", "blood_pressure", "body_temperature", "hrv", "spo2", "stress"
    - date_filter (str): Date in YYYY-MM-DD format (OPTIONAL - if not provided, analyzes all available data)
    - analysis_type (str): "high" or "low" to find patients with concerning values
    
    Returns a list of DISTINCT patients who have readings above/below the threshold, grouped by patient.
    Each patient shows their highest/lowest value and sample readings.
    
    Use this for queries like:
    - "List patients with high glucose readings" (no date needed)
    - "Find patients with low blood pressure" (analyzes all data)
    - "List all patients whose sugar value is high on 16th July 2025" (with specific date)
    - "Which patients had high glucose on a specific date" (returns unique patients, not duplicate readings)
    """
    
    def _run(self, reading_type: str = "glucose", date_filter: Optional[str] = None,
             analysis_type: str = "high") -> str:
        """Analyze readings across multiple patients - ENHANCED WITH OPTIONAL DATES"""
        try:
            with DatabaseManager() as db_manager:
                # Parse date input (optional)
                date_datetime = None
                if date_filter:
                    try:
                        date_datetime = datetime.fromisoformat(date_filter)
                    except ValueError:
                        try:
                            date_datetime = datetime.strptime(date_filter, "%Y-%m-%d")
                        except ValueError:
                            return f"Error: Invalid date format. Use YYYY-MM-DD"
                
                # Get high/low readings for all patients
                result = db_manager.get_high_low_readings(
                    reading_type=reading_type,
                    date_filter=date_datetime,
                    find_type=analysis_type,
                    all_patients=True
                )
                
                if "error" in result:
                    return f"Error: {result['error']}"
                
                return json.dumps(result, indent=2)
                
        except Exception as e:
            logger.error(f"Error in MultiPatientAnalysisTool: {e}")
            return f"Error analyzing multiple patients: {str(e)}"
