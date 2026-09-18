#!/usr/bin/env python3
"""
Simple token validation routes for Revival Medical System
"""

import logging
from fastapi import APIRouter, Depends
from auth.auth import get_current_user, UserContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/validate")
async def validate_token(current_user: UserContext = Depends(get_current_user)):
    """Validate database token and return user info"""
    return {
        "valid": True,
        "user_id": current_user.user_id,
        "role_id": current_user.role_id,
        "role_name": current_user.role_name,
        "can_access_all_patients": current_user.can_access_all_patients,
        "email": current_user.email,
        "full_name": current_user.full_name
    }

@router.get("/me")
async def get_current_user_info(current_user: UserContext = Depends(get_current_user)):
    """Get current user information"""
    return {
        "user_id": current_user.user_id,
        "role_id": current_user.role_id,
        "role_name": current_user.role_name,
        "can_access_all_patients": current_user.can_access_all_patients,
        "email": current_user.email,
        "full_name": current_user.full_name
    }

@router.get("/health")
async def health_check():
    """Health check endpoint for auth service"""
    return {"status": "healthy", "service": "auth_validation"}
