#!/usr/bin/env python3
"""
Plan Tool for Revival Medical System
"""

import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime
from langchain.tools import BaseTool
from dal.database import DatabaseManager

logger = logging.getLogger(__name__)

class PlanTool(BaseTool):
    """Tool for getting patient plan information"""
    name: str = "get_my_plan"
    description: str = """Get current plan details and usage summary for a patient. Use this tool for any plan-related queries.
    
    Parameters:
    - patient_id (int): Patient ID number (optional if patient_name provided)
    - patient_name (str): Patient name (optional if patient_id provided)
    - plan_type (str): Type of plan info - "current", "all", "summary" (default: "current")
    
    Use this tool for queries like:
    - "What is my plan?" / "What's my plan?" / "Show my plan"
    - "My plan details" / "Current plan" / "Treatment plan"
    - "Show my plan details" / "What is my current plan?"
    - "How many consultations do I have left?" (use plan_type="summary")
    - "What are my plan benefits?" / "Plan usage summary"
    - "Show my plan usage" / "Plan details"
    - "When does my plan expire?"
    
    This tool handles all patient plan queries and returns detailed plan information including:
    - Plan name and type, consultation limits and usage, plan benefits and features, remaining consultation counts.
    """
    
    def __init__(self):
        super().__init__()
        # Don't set user_context as instance variable to avoid Pydantic validation issues
    
    def set_user_context(self, user_context):
        """Set user context for role-based access control"""
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, 'user_context', user_context)
    
    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
             plan_type: str = "current") -> str:
        """Get patient plan information with role-based access control"""
        logger.info(f"🔍 PlanTool._run called with patient_id={patient_id}, patient_name={patient_name}, plan_type={plan_type}")
        user_context = getattr(self, 'user_context', None)
        logger.info(f"🔍 User context: {user_context}")
        
        try:
            # Enforce role-based access control
            if user_context and user_context.get('role_id') == 1:  # Patient role
                # Patients can only access their own plan
                patient_id = user_context.get('user_id')
                patient_name = None  # Override any patient_name to enforce access control
                logger.info(f"Patient access: restricting plan query to patient ID {patient_id}")
            elif patient_id is None and patient_name is None:
                # For medical staff, if no patient specified, this might be an error
                return json.dumps({
                    "error": "Please specify a patient ID or patient name for the plan query."
                }, indent=2)
            
            with DatabaseManager() as db_manager:
                if plan_type == "summary":
                    # Get plan usage summary
                    result = db_manager.get_plan_usage_summary(patient_id=patient_id, patient_name=patient_name)
                    
                    if not result.get('has_active_plan'):
                        return json.dumps({
                            "message": "No active plan found for this patient",
                            "has_active_plan": False
                        }, indent=2)
                    
                    return json.dumps({
                        "plan_summary": result,
                        "message": f"Plan usage summary for {result['plan_name']}"
                    }, indent=2)
                
                elif plan_type == "all":
                    # Get all plans (active and inactive)
                    plans = db_manager.get_user_plans(patient_id=patient_id, patient_name=patient_name, active_only=False)
                    
                    if not plans:
                        return json.dumps({
                            "message": "No plans found for this patient",
                            "plans": []
                        }, indent=2)
                    
                    return json.dumps({
                        "plans": plans,
                        "total_plans": len(plans),
                        "message": f"Found {len(plans)} plans for patient"
                    }, indent=2)
                
                else:  # plan_type == "current" or default
                    # Get current active plan
                    current_plan = db_manager.get_current_active_plan(patient_id=patient_id, patient_name=patient_name)
                    
                    if not current_plan:
                        # Try to get the most recent plan
                        all_plans = db_manager.get_user_plans(patient_id=patient_id, patient_name=patient_name, active_only=False)
                        if all_plans:
                            most_recent = all_plans[0]  # Already sorted by purchase date desc
                            return json.dumps({
                                "message": "No currently active plan found. Showing most recent plan:",
                                "plan": most_recent,
                                "status": "inactive"
                            }, indent=2)
                        else:
                            return json.dumps({
                                "message": "No plans found for this patient",
                                "has_plan": False
                            }, indent=2)
                    
                    # Get usage summary for the current plan
                    usage_summary = db_manager.get_plan_usage_summary(patient_id=patient_id, patient_name=patient_name)
                    
                    return json.dumps({
                        "current_plan": current_plan,
                        "usage_summary": usage_summary,
                        "message": f"Current active plan: {current_plan['plan_name']}"
                    }, indent=2)
            
        except Exception as e:
            logger.error(f"Error in PlanTool: {e}")
            return json.dumps({
                "error": f"Failed to retrieve plan information: {str(e)}"
            }, indent=2)
