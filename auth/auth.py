#!/usr/bin/env python3
"""
Cognito Token Authentication and Authorization utilities for Revival Medical System
Verifies Cognito ID tokens directly (JWKS signature check) instead of a DB-stored token.
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import jwt, JWTError
import requests

from dal.models.role import Role

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

# ─────────────────────────────────────────────
# Cognito JWT verification
# ─────────────────────────────────────────────
COGNITO_REGION = os.getenv("COGNITO_REGION")
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID")
COGNITO_APP_CLIENT_ID = os.getenv("COGNITO_APP_CLIENT_ID")
COGNITO_ISSUER = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}"
COGNITO_JWKS_URL = f"{COGNITO_ISSUER}/.well-known/jwks.json"

_jwks_cache: Optional[Dict[str, Any]] = None
_jwks_cache_time: float = 0
_JWKS_CACHE_TTL = 3600  # refetch keys at most once an hour

def _get_jwks() -> Dict[str, Any]:
    """Fetch Cognito's public signing keys, cached for _JWKS_CACHE_TTL seconds."""
    global _jwks_cache, _jwks_cache_time
    now = time.time()
    if _jwks_cache is None or (now - _jwks_cache_time) > _JWKS_CACHE_TTL:
        resp = requests.get(COGNITO_JWKS_URL, timeout=10)
        resp.raise_for_status()
        _jwks_cache = resp.json()
        _jwks_cache_time = now
    return _jwks_cache

def verify_cognito_token(token: str) -> Dict[str, Any]:
    """
    Verify a Cognito ID token's signature, issuer, audience and expiry,
    and return its decoded claims. Raises HTTPException(401) on any failure.
    """
    try:
        jwks = _get_jwks()
        header = jwt.get_unverified_header(token)
        key = next((k for k in jwks["keys"] if k["kid"] == header.get("kid")), None)
        if key is None:
            raise HTTPException(status_code=401, detail="Invalid token: signing key not found")

        claims = jwt.decode(
            token,
            key,
            algorithms=[header.get("alg", "RS256")],
            audience=COGNITO_APP_CLIENT_ID,
            issuer=COGNITO_ISSUER,
        )
        # ID tokens carry token_use="id"; access tokens don't carry our custom claims
        if claims.get("token_use") != "id":
            raise HTTPException(status_code=401, detail="Expected a Cognito ID token")

        return claims

    except JWTError as e:
        logger.warning(f"Cognito token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

class UserContext(BaseModel):
    """User context for authorization"""
    user_id: int
    role_id: int
    role_name: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    token: str
    can_access_all_patients: bool = False



def get_role_name(role_id: int) -> str:
    """Get role name from role ID"""
    role_mapping = {
        Role.PATIENT: "Patient",
        Role.DOCTOR: "Doctor",
        Role.HEALTH_COACH: "Health Coach",
        Role.ADMIN: "Admin",
        Role.DIAGNOSTIC: "Diagnostic",
        Role.VIDEO_UPLOADER: "Video Uploader",
        Role.TRAINER: "Trainer",
        Role.TRACKER: "Tracker",
        Role.CRM_ADMIN: "CRM Admin",
        Role.CRM_EXECUTIVE: "CRM Executive",
        Role.VENDOR: "Vendor",
        Role.ORDER_MANAGER: "Order Manager",
        Role.VIDEO_ADMIN: "Video Admin",
        Role.READ_ONLY: "Read Only"
    }
    return role_mapping.get(role_id, "Unknown")

def determine_access_level(role_id: int) -> bool:
    """Determine if role can access all patients"""
    # Roles that can access all patient data
    privileged_roles = [
        Role.DOCTOR,
        Role.HEALTH_COACH,
        Role.ADMIN,
        Role.DIAGNOSTIC,
        Role.CRM_ADMIN,
        Role.CRM_EXECUTIVE
    ]
    
    return role_id in privileged_roles

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UserContext:
    """
    Get current authenticated user by verifying the Cognito ID token
    and building UserContext directly from its claims.
    """
    try:
        token = credentials.credentials.strip()
        if not token:
            raise HTTPException(status_code=401, detail="Token is required")

        claims = verify_cognito_token(token)

        try:
            user_id = int(claims.get("custom:id"))
            role_id = int(claims.get("custom:roleId"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=401, detail="Token is missing required user/role claims")

        role_name = claims.get("custom:roleName") or get_role_name(role_id)
        full_name = f"{claims.get('given_name') or ''} {claims.get('family_name') or ''}".strip()
        can_access_all = determine_access_level(role_id)

        logger.debug(f"Authenticated user_id={user_id} role={role_name}")

        return UserContext(
            user_id=user_id,
            role_id=role_id,
            role_name=role_name,
            email=claims.get("email"),
            full_name=full_name,
            token=token,
            can_access_all_patients=can_access_all
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_current_user: {e}")
        raise HTTPException(status_code=500, detail="Authentication error")

def require_patient_access(requested_patient_id: Optional[int], current_user: UserContext) -> bool:
    """Check if current user can access requested patient data"""
    try:
        # If user can access all patients, allow access
        if current_user.can_access_all_patients:
            return True
        
        # If user is a patient, they can only access their own data
        if current_user.role_id == Role.PATIENT:
            if requested_patient_id is None:
                # If no specific patient requested, default to current user
                return True
            elif requested_patient_id == current_user.user_id:
                # Patient accessing their own data
                return True
            else:
                # Patient trying to access another patient's data
                raise HTTPException(
                    status_code=403, 
                    detail="Patients can only access their own medical data"
                )
        
        # Default deny for any other scenario
        raise HTTPException(status_code=403, detail="Access denied")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in require_patient_access: {e}")
        raise HTTPException(status_code=500, detail="Authorization error")

def get_authorized_patient_id(requested_patient_id: Optional[int], current_user: UserContext) -> int:
    """Get the authorized patient ID based on user role and request"""
    try:
        # If user can access all patients and a specific patient is requested
        if current_user.can_access_all_patients and requested_patient_id:
            return requested_patient_id
        
        # If user can access all patients but no specific patient requested, return None to indicate "all patients"
        elif current_user.can_access_all_patients and not requested_patient_id:
            return None
        
        # If user is a patient, always return their own ID
        elif current_user.role_id == Role.PATIENT:
            return current_user.user_id
        
        # Default fallback
        else:
            raise HTTPException(status_code=403, detail="Cannot determine authorized patient access")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_authorized_patient_id: {e}")
        raise HTTPException(status_code=500, detail="Authorization error")

# Role-based decorators
def require_roles(*allowed_roles):
    """Decorator to require specific roles"""
    def decorator(func):
        async def wrapper(*args, current_user: UserContext = Depends(get_current_user), **kwargs):
            if current_user.role_id not in allowed_roles:
                role_names = [str(role) for role in allowed_roles]
                raise HTTPException(
                    status_code=403, 
                    detail=f"Access denied. Required roles: {', '.join(role_names)}"
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator

def require_admin(current_user: UserContext = Depends(get_current_user)) -> UserContext:
    """Require admin role"""
    if current_user.role_id != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

def require_medical_staff(current_user: UserContext = Depends(get_current_user)) -> UserContext:
    """Require medical staff roles (Doctor, Health Coach, Diagnostic)"""
    medical_roles = [Role.DOCTOR, Role.HEALTH_COACH, Role.DIAGNOSTIC]
    if current_user.role_id not in medical_roles:
        raise HTTPException(status_code=403, detail="Medical staff access required")
    return current_user