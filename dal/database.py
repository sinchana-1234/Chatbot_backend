#!/usr/bin/env python3
"""
Database models for Revival Medical System
Main database manager and connection handling - Service Layer Pattern
"""

import os
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
from .models.base import Base
from sqlalchemy.orm import sessionmaker, relationship, Session
from dotenv import load_dotenv

# Import all model classes that might be needed
from .models.users import Users
from .models.glucose_readings import GlucoseReadings
from .models.activity_readings import ActivityReadings
from .models.blood_pressure_readings import BloodPressureReadings
from .models.body_temperature_readings import BodyTemperatureReadings
from .models.hrv_readings import HrvReadings
from .models.spo2_readings import Spo2Readings
from .models.stress_readings import StressReadings
from .models.medications import Medications
from .models.sleep_readings_details import SleepReadingsDetails
from .models.foodlog import Foodlog
from .models.protocol import Protocol

# Import services
from .services.medical_readings_service import MedicalReadingsService
from .services.medications_service import MedicationsService
from .services.foodlog_service import FoodlogService
from .services.protocol_service import ProtocolService
from .services.plan_service import PlanService
from .services.patient_doctor_mapping_service import PatientDoctorMappingService

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Database setup
# Base = declarative_base()
engine = None
SessionLocal = None

def get_database_url() -> str:
    return os.getenv("MYSQL_DATABASE_URL")

def init_database():
    """Initialize database connection and create tables"""
    global engine, SessionLocal
    
    try:
        database_url = get_database_url()
        logger.info(f"Connecting to database: {database_url}")
        
        # Apply SSL only for managed cloud MySQL (Azure, AWS RDS).
        # Local/QA MySQL at internal IPs typically doesn't support SSL,
        # AND the cert path /certs/... only exists inside the container.
        # On Windows local dev this caused "No such file or directory".
        needs_ssl = database_url and any(host in database_url for host in [
            ".database.azure.com",
            ".rds.amazonaws.com",
        ])
        # Timeouts to prevent any MySQL operation from hanging indefinitely.
        #
        # Root cause of 524 on /auth/validate:
        # Docker container connects to RDS → MySQL sees source as Docker bridge
        # IP (172.17.0.1) → MySQL does reverse DNS lookup on that IP →
        # lookup fails (172.17.0.1 has no DNS entry) → connection hangs.
        # Permanent fix: set skip_name_resolve=1 in RDS Parameter Group (Pradeep).
        # Code fix: these timeouts ensure hung operations fail fast instead of
        # hanging 120s and hitting Cloudflare 524.
        #
        # connect_timeout: max seconds to establish TCP connection to MySQL
        # read_timeout:    max seconds to wait for query result
        # write_timeout:   max seconds to wait for write to complete
        connect_args = {
            "connect_timeout": 5,
            "read_timeout":    30,
            "write_timeout":   30,
        }
        if needs_ssl:
            connect_args["ssl"] = {"ca": "/certs/DigiCertGlobalRootG2.crt.pem"}

        engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600,
            pool_timeout=30,   # back to 30s — gives time for connections to free up under load
            connect_args=connect_args,
        )
        
        # Create session factory
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        logger.info("Database connection established successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

def get_db() -> Session:
    """Get database session"""
    if SessionLocal is None:
        raise Exception("Database not initialized. Call init_database() first.")
    
    return SessionLocal()

class DatabaseManager:
    """Main database manager using service layer pattern"""
    
    def __init__(self, auto_init: bool = True):
        self.db = None
        self._medical_readings_service = None
        self._medications_service = None
        self._foodlog_service = None
        self._protocol_service = None
        self._plan_service = None
        self._patient_doctor_mapping_service = None
        
        if auto_init and engine is None:
            # Only initialise once — engine and SessionLocal are module-level globals.
            # Calling init_database() on every request creates a brand-new connection
            # pool each time, which exhausts MySQL's max_connect_errors limit and
            # causes intermittent 401s and page-load hangs in production.
            try:
                init_database()
            except Exception as e:
                logger.warning(f"Database initialization failed: {e}")
        
        self._get_session()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup"""
        self.close()
    
    def close(self):
        """Close database session"""
        if self.db:
            try:
                self.db.close()
                logger.debug("Database session closed")
            except Exception as e:
                logger.error(f"Error closing database session: {e}")
            finally:
                self.db = None
                # Reset services
                self._medical_readings_service = None
                self._medications_service = None
                self._foodlog_service = None
                self._protocol_service = None
                self._plan_service = None
    
    def _get_session(self):
        """Get a fresh database session"""
        try:
            if SessionLocal is None:
                raise Exception("Database not initialized. Call init_database() first.")
            
            # Close existing session if any
            if self.db:
                try:
                    self.db.close()
                except Exception:
                    pass
            
            # Create new session
            self.db = SessionLocal()
            
            # Reset services to use new session
            self._medical_readings_service = None
            self._medications_service = None
            self._foodlog_service = None
            self._protocol_service = None
            
        except Exception as e:
            logger.error(f"Failed to create database session: {e}")
            self.db = None
    
    def _handle_db_error(self, error):
        """Handle database errors by rolling back and creating new session"""
        logger.error(f"Database error: {error}")
        if self.db:
            try:
                self.db.rollback()
                self.db.close()
            except Exception:
                pass
            # Create fresh session
            self._get_session()
    
    # Service property accessors
    @property
    def medical_readings_service(self) -> Optional[MedicalReadingsService]:
        """Get medical readings service instance"""
        if not self._medical_readings_service and self.db:
            self._medical_readings_service = MedicalReadingsService(self.db)
        return self._medical_readings_service
    
    @property
    def medications_service(self) -> Optional[MedicationsService]:
        """Get medications service instance"""
        if not self._medications_service and self.db:
            self._medications_service = MedicationsService(self.db)
        return self._medications_service
    
    @property
    def foodlog_service(self) -> Optional[FoodlogService]:
        """Get foodlog service instance"""
        if not self._foodlog_service and self.db:
            self._foodlog_service = FoodlogService(self.db)
        return self._foodlog_service
    
    @property
    def protocol_service(self) -> Optional[ProtocolService]:
        """Get protocol service instance"""
        if not self._protocol_service and self.db:
            self._protocol_service = ProtocolService(self.db)
        return self._protocol_service
    
    @property
    def plan_service(self) -> Optional[PlanService]:
        """Get plan service instance"""
        if not self._plan_service and self.db:
            self._plan_service = PlanService(self.db)
        return self._plan_service
    
    @property
    def patient_doctor_mapping_service(self) -> Optional[PatientDoctorMappingService]:
        """Get patient doctor mapping service instance"""
        if not self._patient_doctor_mapping_service and self.db:
            self._patient_doctor_mapping_service = PatientDoctorMappingService(self.db)
        return self._patient_doctor_mapping_service
    
    # Delegate methods to services
    def get_specific_reading_value(self, **kwargs) -> Dict[str, Any]:
        """Delegate to medical readings service"""
        if not self.db:
            self._get_session()
        if not self.db:
            return {"error": "Database connection failed"}
        
        try:
            service = self.medical_readings_service
            if not service:
                return {"error": "Medical readings service unavailable"}
            return service.get_specific_reading_value(**kwargs)
        except Exception as e:
            self._handle_db_error(e)
            return {"error": f"Database error: {str(e)}"}
    
    def get_high_low_readings(self, **kwargs) -> Dict[str, Any]:
        """Delegate to medical readings service"""
        if not self.db:
            self._get_session()
        if not self.db:
            return {"error": "Database connection failed"}
        
        try:
            service = self.medical_readings_service
            if not service:
                return {"error": "Medical readings service unavailable"}
            return service.get_high_low_readings(**kwargs)
        except Exception as e:
            self._handle_db_error(e)
            return {"error": f"Database error: {str(e)}"}
    
    def get_medications(self, **kwargs) -> Dict[str, Any]:
        """Delegate to medications service"""
        if not self.db:
            self._get_session()
        if not self.db:
            return {"error": "Database connection failed"}
        
        try:
            service = self.medications_service
            if not service:
                return {"error": "Medications service unavailable"}
            return service.get_medications(**kwargs)
        except Exception as e:
            self._handle_db_error(e)
            return {"error": f"Database error: {str(e)}"}
    
    def get_foodlog(self, **kwargs) -> Dict[str, Any]:
        """Delegate to foodlog service"""
        if not self.db:
            self._get_session()
        if not self.db:
            return {"error": "Database connection failed"}
        
        try:
            service = self.foodlog_service
            if not service:
                return {"error": "Foodlog service unavailable"}
            return service.get_foodlog(**kwargs)
        except Exception as e:
            self._handle_db_error(e)
            return {"error": f"Database error: {str(e)}"}
    
    def get_protocols(self, **kwargs) -> Dict[str, Any]:
        """Delegate to protocol service"""
        if not self.db:
            self._get_session()
        if not self.db:
            return {"error": "Database connection failed"}
        
        try:
            service = self.protocol_service
            if not service:
                return {"error": "Protocol service unavailable"}
            return service.get_protocols(**kwargs)
        except Exception as e:
            self._handle_db_error(e)
            return {"error": f"Database error: {str(e)}"}
    
    def get_user_plans(self, **kwargs) -> List[Dict[str, Any]]:
        """Delegate to plan service"""
        if not self.db:
            self._get_session()
        if not self.db:
            return []
        
        try:
            service = self.plan_service
            if not service:
                return []
            return service.get_user_plans(**kwargs)
        except Exception as e:
            self._handle_db_error(e)
            return []
    
    def get_current_active_plan(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Delegate to plan service"""
        if not self.db:
            self._get_session()
        if not self.db:
            return None
        
        try:
            service = self.plan_service
            if not service:
                return None
            return service.get_current_active_plan(**kwargs)
        except Exception as e:
            self._handle_db_error(e)
            return None
    
    def get_plan_usage_summary(self, **kwargs) -> Dict[str, Any]:
        """Delegate to plan service"""
        if not self.db:
            self._get_session()
        if not self.db:
            return {"error": "Database connection failed"}
        
        try:
            service = self.plan_service
            if not service:
                return {"error": "Plan service unavailable"}
            return service.get_plan_usage_summary(**kwargs)
        except Exception as e:
            self._handle_db_error(e)
            return {"error": f"Database error: {str(e)}"}
    
    def get_users(self, user_id: Optional[int] = None, mobile_number: Optional[str] = None, email: Optional[str] = None) -> List:
        """Get users with filters"""
        if not self.db:
            self._get_session()
            
        if not self.db:
            return []
            
        try:
            query = self.db.query(Users)
            
            if user_id:
                query = query.filter(Users.id == user_id)
            if mobile_number:
                query = query.filter(Users.mobile_number == mobile_number)
            if email:
                query = query.filter(Users.email.ilike(f"%{email}%"))
            
            result = query.all()
            return result
            
        except Exception as e:
            self._handle_db_error(e)
            return []
    
    # Patient Doctor Mapping delegate methods
    def get_patient_doctors(self, **kwargs) -> List[Dict[str, Any]]:
        """Delegate to patient doctor mapping service"""
        if not self.db:
            self._get_session()
        if not self.db:
            return []
        
        try:
            service = self.patient_doctor_mapping_service
            if service:
                return service.get_patient_doctors(**kwargs)
            return []
        except Exception as e:
            self._handle_db_error(e)
            return []
    
    def get_doctor_patients(self, **kwargs) -> List[Dict[str, Any]]:
        """Delegate to patient doctor mapping service"""
        if not self.db:
            self._get_session()
        if not self.db:
            return []
        
        try:
            service = self.patient_doctor_mapping_service
            if service:
                return service.get_doctor_patients(**kwargs)
            return []
        except Exception as e:
            self._handle_db_error(e)
            return []
    
    def get_primary_doctor(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Delegate to patient doctor mapping service"""
        if not self.db:
            self._get_session()
        if not self.db:
            return None
        
        try:
            service = self.patient_doctor_mapping_service
            if service:
                return service.get_primary_doctor(**kwargs)
            return None
        except Exception as e:
            self._handle_db_error(e)
            return None
    
    def check_doctor_patient_access(self, **kwargs) -> bool:
        """Delegate to patient doctor mapping service"""
        if not self.db:
            self._get_session()
        if not self.db:
            return False
        
        try:
            service = self.patient_doctor_mapping_service
            if service:
                return service.check_doctor_patient_access(**kwargs)
            return False
        except Exception as e:
            self._handle_db_error(e)
            return False
    def get_medical_readings(self, **kwargs) -> Dict[str, Any]:
        """Delegate to medical readings service for full-day data"""
        if not self.db:
            self._get_session()
        if not self.db:
            return {"error": "Database connection failed"}

        try:
            service = self.medical_readings_service
            if not service:
                return {"error": "Medical readings service unavailable"}
            return service.get_medical_readings(**kwargs)
        except Exception as e:
            self._handle_db_error(e)
            return {"error": f"Database error: {str(e)}"}


    
def get_db_manager() -> DatabaseManager:
    """Get a DatabaseManager instance with connection handling"""
    return DatabaseManager(auto_init=True)