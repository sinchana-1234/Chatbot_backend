#!/usr/bin/env python3
"""
User Profile Tool for Revival Medical System
Combines user profile data with plan information
"""

import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime, date
from langchain.tools import BaseTool
from dal.database import DatabaseManager

logger = logging.getLogger(__name__)

class UserProfileTool(BaseTool):
    """Tool for getting complete user profile including personal info and plan details"""
    name: str = "get_user_profile"
    description: str = """Get complete user profile including personal information and plan details with role-based access control.
    
    Parameters:
    - patient_id (int): Patient ID (optional for patient role, required for staff queries)
    - patient_name (str): Patient name (optional, alternative to patient_id for staff)
    - include_plans (bool): Whether to include plan information (default: True)
    - active_plans_only (bool): Show only active plans (default: True)
    
    Use this tool for queries like:
    - "Show my profile" → (patient role)
    - "What's my age/sex/details?" → (patient role)
    - "Show my profile with plan details" → (patient role)
    - "Profile for patient 132" → patient_id=132 (staff role)
    - "Show John's profile" → patient_name="John" (staff role)
    - "Get patient profile with all plans" → active_plans_only=False (staff role)
    
    This tool returns comprehensive profile information including:
    - Personal details (name, age, sex, contact info)
    - Current active plans and their details
    - Plan usage and remaining benefits
    - Profile status and registration info
    """
    
    def __init__(self):
        super().__init__()
        # Don't set user_context as instance variable to avoid Pydantic validation issues
    
    def set_user_context(self, user_context):
        """Set user context for role-based access control"""
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, 'user_context', user_context)
    
    def _calculate_age(self, dob):
        """Calculate age from date of birth"""
        if not dob:
            return None
        
        if isinstance(dob, str):
            try:
                dob = datetime.strptime(dob, '%Y-%m-%d').date()
            except:
                return None
        elif isinstance(dob, datetime):
            dob = dob.date()
        
        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age
    
    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
             include_plans: bool = True, active_plans_only: bool = True) -> str:
        """Execute the user profile query with role-based access control"""
        logger.info(f"🔍 UserProfileTool._run called with patient_id={patient_id}, patient_name={patient_name}, include_plans={include_plans}, active_plans_only={active_plans_only}")
        user_context = getattr(self, 'user_context', None)
        logger.info(f"🔍 User context: {user_context}")
        
        try:
            # Enforce role-based access control
            if user_context and user_context.get('role_id') == 1:  # Patient role
                # Patients can only query their own information
                patient_id = user_context.get('user_id')
                logger.info(f"Patient access: restricting query to patient ID {patient_id}")
                
            elif not user_context or user_context.get('role_id') != 1:  # Medical staff
                # Medical staff can query any patient information.
                # No patient specified at all → default to the logged-in staff
                # member's own profile (mirrors "my_patients"/"my_doctor" self-
                # referencing pattern elsewhere in this codebase).
                if not patient_id and not patient_name and user_context:
                    patient_id = user_context.get('user_id')
                if not patient_id and patient_name:
                    # Try to find patient by name
                    with DatabaseManager() as db_manager:
                        users = db_manager.get_users()
                        matching_users = [
                            u for u in users 
                            if patient_name.lower() in f"{u.first_name or ''} {u.last_name or ''}".lower()
                            and u.role_id == 1  # Only patients
                        ]
                        
                        if not matching_users:
                            return json.dumps({
                                "error": f"No patient found with name containing '{patient_name}'",
                                "suggestion": "Try using exact patient name or patient ID"
                            }, indent=2)
                        
                        if len(matching_users) > 1:
                            return json.dumps({
                                "error": f"Multiple patients found with name containing '{patient_name}'",
                                "matching_patients": [
                                    {
                                        "id": u.id, 
                                        "name": f"{u.first_name or ''} {u.last_name or ''}".strip(),
                                        "email": u.email
                                    } for u in matching_users
                                ],
                                "suggestion": "Please specify exact patient ID or more specific name"
                            }, indent=2)
                        
                        patient_id = matching_users[0].id
                
                if not patient_id:
                    return json.dumps({
                        "error": "patient_id or patient_name is required for staff queries"
                    }, indent=2)
            
            with DatabaseManager() as db_manager:
                # Get user details
                user_data = db_manager.get_users(user_id=patient_id)
                if not user_data:
                    return json.dumps({
                        "error": f"User with ID {patient_id} not found"
                    }, indent=2)
                
                user = user_data[0]
                
                # Calculate age from date of birth
                age = self._calculate_age(user.dob)
                
                # Prepare profile information
                profile = {
                    "user_id": user.id,
                    "personal_info": {
                        "first_name": user.first_name,
                        "last_name": user.last_name,
                        "full_name": f"{user.first_name or ''} {user.last_name or ''}".strip(),
                        "email": user.email,
                        "mobile_number": user.mobile_number,
                        "date_of_birth": user.dob.isoformat() if user.dob else None,
                        "age": age,
                        "sex": user.sex,
                        "address": user.address,
                        "city": user.city,
                        "state": user.state,
                        "zipcode": user.zipcode
                    },
                    "account_info": {
                        "role_id": user.role_id,
                        "status": user.status,
                        "created": user.created.isoformat() if user.created else None,
                        "updated": user.updated.isoformat() if user.updated else None,
                        "customer_id": user.customer_id,
                        "profile": user.profile
                    }
                }
                
                # Add plan information if requested
                if include_plans:
                    if active_plans_only:
                        # Get current active plan
                        active_plan = db_manager.get_current_active_plan(patient_id=patient_id)
                        if active_plan:
                            profile["active_plan"] = active_plan
                        else:
                            profile["active_plan"] = None
                            profile["plan_message"] = "No active plan found"
                        
                        # Get plan usage summary
                        usage_summary = db_manager.get_plan_usage_summary(patient_id=patient_id)
                        if usage_summary:
                            profile["plan_usage"] = usage_summary
                    else:
                        # Get all plans
                        all_plans = db_manager.get_user_plans(patient_id=patient_id)
                        profile["all_plans"] = all_plans
                        profile["total_plans"] = len(all_plans)
                        
                        # Still get usage summary for current plan
                        usage_summary = db_manager.get_plan_usage_summary(patient_id=patient_id)
                        if usage_summary:
                            profile["plan_usage"] = usage_summary
                
                # Add summary message
                plan_info = ""
                if include_plans and profile.get("active_plan"):
                    plan_name = profile["active_plan"].get("plan_name", "Unknown")
                    plan_price = profile["active_plan"].get("plan_price", 0)
                    plan_info = f" with active plan: {plan_name} (₹{plan_price:,})"
                
                profile["summary"] = f"Profile for {profile['personal_info']['full_name']} (ID: {patient_id}){plan_info}"
                
                return json.dumps(profile, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Error in UserProfileTool: {e}")
            return json.dumps({
                "error": f"Database error: {str(e)}",
                "patient_id": patient_id
            }, indent=2)
    
    async def _arun(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
                    include_plans: bool = True, active_plans_only: bool = True) -> str:
        """Async version of the run method"""
        return self._run(patient_id, patient_name, include_plans, active_plans_only)
