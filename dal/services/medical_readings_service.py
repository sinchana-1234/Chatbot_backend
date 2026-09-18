#!/usr/bin/env python3
"""
Medical readings service for glucose, blood pressure, temperature, etc.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import sqlalchemy

from .base_service import BaseService

logger = logging.getLogger(__name__)

class MedicalReadingsService(BaseService):
    """Service for handling medical readings operations"""
    
    def __init__(self, db_session: Session):
        super().__init__(db_session)
        self.model_map = self._get_model_map()
    
    def _get_model_map(self):
        """Map reading types to their corresponding models"""
        from ..models.glucose_readings import GlucoseReadings
        from ..models.blood_pressure_readings import BloodPressureReadings
        from ..models.body_temperature_readings import BodyTemperatureReadings
        from ..models.hrv_readings import HrvReadings
        from ..models.spo2_readings import Spo2Readings
        from ..models.stress_readings import StressReadings
        from ..models.sleep_readings_details import SleepReadingsDetails
        from ..models.activity_readings import ActivityReadings
        
        return {
            "glucose": GlucoseReadings,
            "blood_pressure": BloodPressureReadings,
            "body_temperature": BodyTemperatureReadings,
            "hrv": HrvReadings,
            "spo2": Spo2Readings,
            "stress": StressReadings,
            "sleep": SleepReadingsDetails,
            "activity": ActivityReadings
        }
    
    def get_specific_reading_value(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
                                 reading_type: str = "glucose", specific_time: Optional[datetime] = None,
                                 date_filter: Optional[datetime] = None, time_range: Optional[str] = None,
                                 analysis_type: str = "specific", limit: int = 5, month_filter: bool = False) -> Dict[str, Any]:
        """Get specific reading value with time/date filters"""
        try:
            # Find patient ID
            patient_id = self.find_patient_by_name_or_id(patient_id, patient_name)
            if not patient_id:
                return {"error": "Patient not found"}
            
            if reading_type not in self.model_map:
                return {"error": f"Invalid reading type: {reading_type}"}
            
            model = self.model_map[reading_type]
            timestamp_field = "date" if reading_type == "sleep" else "timestamp"
            
            query = self.db.query(model).filter(model.patient_id == patient_id)
            
            # Apply time filters
            if specific_time:
                time_window = timedelta(hours=1)
                query = query.filter(
                    getattr(model, timestamp_field) >= specific_time - time_window,
                    getattr(model, timestamp_field) <= specific_time + time_window
                )
            elif date_filter:
                if month_filter:
                    # Filter for entire month
                    from calendar import monthrange
                    year = date_filter.year
                    month = date_filter.month
                    last_day = monthrange(year, month)[1]
                    month_start = datetime(year, month, 1)
                    month_end = datetime(year, month, last_day, 23, 59, 59)
                    query = query.filter(
                        getattr(model, timestamp_field) >= month_start,
                        getattr(model, timestamp_field) <= month_end
                    )
                else:
                    # Filter for specific day
                    query = query.filter(
                        getattr(model, timestamp_field) >= date_filter,
                        getattr(model, timestamp_field) < date_filter + timedelta(days=1)
                    )
                
                # Apply time range filter for non-sleep data
                if time_range and reading_type != "sleep":
                    if time_range.lower() in ["morning", "am"]:
                        query = query.filter(
                            getattr(model, timestamp_field).cast(sqlalchemy.Time) >= datetime.strptime("06:00", "%H:%M").time(),
                            getattr(model, timestamp_field).cast(sqlalchemy.Time) <= datetime.strptime("12:00", "%H:%M").time()
                        )
                    elif time_range.lower() in ["night", "evening", "pm"]:
                        query = query.filter(
                            getattr(model, timestamp_field).cast(sqlalchemy.Time) >= datetime.strptime("18:00", "%H:%M").time(),
                            getattr(model, timestamp_field).cast(sqlalchemy.Time) <= datetime.strptime("23:59", "%H:%M").time()
                        )
            
            # Handle different analysis types
            if analysis_type == "highest":
                # Order by value descending to get highest values
                value_field = self._get_value_field(reading_type)
                query = query.order_by(getattr(model, value_field).desc())
                readings = query.limit(limit).all()
            elif analysis_type == "lowest":
                # Order by value ascending to get lowest values
                value_field = self._get_value_field(reading_type)
                query = query.order_by(getattr(model, value_field).asc())
                readings = query.limit(limit).all()
            elif analysis_type == "average":
                value_field = self._get_value_field(reading_type)

                avg_value = query.with_entities(
                    sqlalchemy.func.avg(getattr(model, value_field))
                ).scalar()

                return {
                    "patient_id": patient_id,
                    "reading_type": reading_type,
                    "analysis_type": "average",
                    "average_value": round(float(avg_value), 2) if avg_value is not None else None
                }
            else:
                # Default: order by timestamp descending
                query = query.order_by(getattr(model, timestamp_field).desc())
                if analysis_type == "specific" and reading_type != "sleep":
                    readings = query.limit(1).all()  # Only one result for specific (except sleep)
                elif reading_type == "sleep":
                    readings = query.all()  # Get ALL sleep records for proper calculation
                
                else:
                    readings = query.limit(limit).all()
            
            # Special handling for sleep data
            if reading_type == "sleep":
                return self._process_sleep_data(readings, patient_id, date_filter, analysis_type)
            
            # Process non-sleep readings
            return self._process_standard_readings(readings, patient_id, reading_type, analysis_type)
            
        except Exception as e:
            logger.error(f"Error getting specific reading value: {e}")
            return {"error": f"Database error: {str(e)}"}
    
    def _process_sleep_data(self, readings: List, patient_id: int, date_filter: Optional[datetime], analysis_type: str = "specific") -> Dict[str, Any]:
        """Process sleep data with total calculation"""
        total_sleep_minutes = 0
        reading_list = []
        sleep_breakdown = {
            "deep_sleep": 0,     # level = 0
            "light_sleep": 0,    # level = 1
            "rem_sleep": 0,      # level = 2
            "awake": 0           # level = 3
        }
        
        for reading in readings:
            sleep_value = getattr(reading, 'value', 0) or 0
            sleep_type = getattr(reading, 'sleep_type', 'unknown') or 'unknown'
            level = getattr(reading, 'level', None)
            
            # Each row = 1 minute of recorded sleep at this level.
            # Count rows (not value) to get true minutes per stage.
            if level == 0:
                sleep_breakdown["deep_sleep"] += 1
                total_sleep_minutes += 1
            elif level == 1:
                sleep_breakdown["light_sleep"] += 1
                total_sleep_minutes += 1
            elif level == 2:
                sleep_breakdown["rem_sleep"] += 1
                total_sleep_minutes += 1
            elif level == 3:
                sleep_breakdown["awake"] += 1
                # awake minutes are tracked but NOT added to total_sleep_minutes
            
            reading_dict = {
                "date": reading.date.isoformat() if hasattr(reading, 'date') and reading.date else None,
                "sleep_type": sleep_type,
                "value": sleep_value,
                "level": level,
                "level_description": self._get_sleep_level_description(level)
            }
            reading_list.append(reading_dict)
        
        total_sleep_hours = total_sleep_minutes / 60.0
        hours = int(total_sleep_hours)
        remaining_minutes = int((total_sleep_hours - hours) * 60)
        
        return {
            "patient_id": patient_id,
            "reading_type": "sleep",
            "date_filter": date_filter.date().isoformat() if date_filter else None,
            "total_sleep_records": len(reading_list),
            "total_sleep_minutes": total_sleep_minutes,
            "total_sleep_hours": total_sleep_hours,
            "total_sleep_duration": f"{hours} hours and {remaining_minutes} minutes" if remaining_minutes > 0 else f"{hours} hours",
            "sleep_breakdown": {
                "deep_sleep_minutes": sleep_breakdown["deep_sleep"],
                "light_sleep_minutes": sleep_breakdown["light_sleep"], 
                "rem_sleep_minutes": sleep_breakdown["rem_sleep"],
                "awake_minutes": sleep_breakdown["awake"],
                "deep_sleep_hours": round(sleep_breakdown["deep_sleep"] / 60.0, 2),
                "light_sleep_hours": round(sleep_breakdown["light_sleep"] / 60.0, 2),
                "rem_sleep_hours": round(sleep_breakdown["rem_sleep"] / 60.0, 2),
                "awake_hours": round(sleep_breakdown["awake"] / 60.0, 2)
            },
            #"individual_readings": reading_list,
            "summary": f"Total sleep time: {hours} hours and {remaining_minutes} minutes from {len(reading_list)} sleep records (deep, light, and REM sleep, excluding {sleep_breakdown['awake']} minutes awake time)"
        }
    
    def _get_sleep_level_description(self, level: Optional[int]) -> str:
        """Get human-readable description for sleep level"""
        if level is None:
            return "unknown"
        elif level == 0:
            return "deep sleep"
        elif level == 1:
            return "light sleep"
        elif level == 2:
            return "REM sleep"
        elif level == 3:
            return "awake"
        else:
            return f"unknown level {level}"
    
    def _process_standard_readings(self, readings: List, patient_id: int, reading_type: str, analysis_type: str = "specific") -> Dict[str, Any]:
        """Process standard medical readings"""
        reading_list = []
        for reading in readings:
            reading_dict = {
                "timestamp": reading.timestamp.isoformat() if hasattr(reading, 'timestamp') and reading.timestamp else None,
                "date": reading.date.isoformat() if hasattr(reading, 'date') and reading.date else None
            }
            
            # Add value fields based on model type
            if hasattr(reading, 'value') and reading.value is not None:
                reading_dict["value"] = reading.value
            elif hasattr(reading, 'temperature') and reading.temperature is not None:
                reading_dict["temperature"] = reading.temperature
            elif hasattr(reading, 'systolic') and reading.systolic is not None:
                reading_dict["systolic"] = reading.systolic
                reading_dict["diastolic"] = reading.diastolic
            
            reading_list.append(reading_dict)
        
        return {
            "patient_id": patient_id,
            "reading_type": reading_type,
            "analysis_type": analysis_type,
            "readings": reading_list,
            "count": len(reading_list),
            "message": f"Found {len(reading_list)} {analysis_type} {reading_type} readings" if analysis_type in ["highest", "lowest"] else f"Found {len(reading_list)} {reading_type} readings"
        }
    
    def get_high_low_readings(self, reading_type: str = "glucose", date_filter: Optional[datetime] = None,
                            find_type: str = "high", all_patients: bool = False) -> Dict[str, Any]:
        """Get highest/lowest readings for all patients with distinct patient grouping"""
        try:
            from ..models.users import Users
            
            if reading_type not in self.model_map:
                return {"error": f"Invalid reading type: {reading_type}. Available types: {list(self.model_map.keys())}"}
            
            # Remove models that don't support high/low analysis
            analysis_models = {k: v for k, v in self.model_map.items() if k not in ["sleep", "activity"]}
            if reading_type not in analysis_models:
                return {"error": f"Reading type {reading_type} doesn't support high/low analysis"}
            
            model = analysis_models[reading_type]
            timestamp_field = "timestamp"
            
            thresholds = {
                "glucose": {"high": 180, "low": 70},
                "blood_pressure": {"high": 140, "low": 90},
                "body_temperature": {"high": 100.4, "low": 96.0},
                "hrv": {"high": 50, "low": 20},
                "spo2": {"high": 100, "low": 90},
                "stress": {"high": 80, "low": 20}
            }
            
            query = self.db.query(model, Users).join(Users, model.patient_id == Users.id)
            
            if date_filter:
                query = query.filter(
                    getattr(model, timestamp_field) >= date_filter,
                    getattr(model, timestamp_field) < date_filter + timedelta(days=1)
                )
            
            if find_type in ["high", "low"]:
                threshold = thresholds[reading_type][find_type]
                value_field = self._get_value_field(reading_type)
                
                if find_type == "high":
                    query = query.filter(getattr(model, value_field) > threshold)
                else:
                    query = query.filter(getattr(model, value_field) < threshold)
            
            results = query.order_by(getattr(model, timestamp_field).desc()).all()
            
            # Group by patient
            distinct_patients = self._group_readings_by_patient(results, reading_type, find_type)
            
            return {
                "reading_type": reading_type,
                "find_type": find_type,
                "threshold": thresholds[reading_type][find_type],
                "date_filter": date_filter.isoformat() if date_filter else "All dates",
                "distinct_patients": distinct_patients,
                "total_patients": len(distinct_patients),
                "total_readings": sum(p["total_readings"] for p in distinct_patients),
                "message": f"Found {len(distinct_patients)} distinct patients with {find_type} {reading_type} readings"
            }
            
        except Exception as e:
            logger.error(f"Error getting high/low readings: {e}")
            return {"error": f"Database error: {str(e)}"}
    
    def _get_value_field(self, reading_type: str) -> str:
        """Get the value field name for a reading type"""
        if reading_type == "blood_pressure":
            return "systolic"
        elif reading_type == "body_temperature":
            return "temperature"
        else:
            return "value"
    
    def _group_readings_by_patient(self, results: List, reading_type: str, find_type: str) -> List[Dict]:
        """Group readings by patient for distinct patient analysis"""
        patient_groups = {}
        
        for reading, user in results:
            patient_id = user.id
            patient_name = f"{user.first_name} {user.last_name}"
            
            value_field = self._get_value_field(reading_type)
            value = getattr(reading, value_field)
            
            additional_info = {}
            if reading_type == "blood_pressure":
                additional_info = {"diastolic": reading.diastolic}
            
            if patient_id not in patient_groups:
                patient_groups[patient_id] = {
                    "patient_id": patient_id,
                    "patient_name": patient_name,
                    "reading_type": reading_type,
                    "readings": [],
                    "highest_value": value,
                    "lowest_value": value,
                    "total_readings": 0
                }
            
            reading_dict = {
                "timestamp": reading.timestamp.isoformat() if reading.timestamp else None,
                "value": value
            }
            reading_dict.update(additional_info)
            
            patient_groups[patient_id]["readings"].append(reading_dict)
            patient_groups[patient_id]["total_readings"] += 1
            
            if value > patient_groups[patient_id]["highest_value"]:
                patient_groups[patient_id]["highest_value"] = value
            if value < patient_groups[patient_id]["lowest_value"]:
                patient_groups[patient_id]["lowest_value"] = value
        
        # Convert to list and limit readings shown
        distinct_patients = []
        for patient_data in patient_groups.values():
            patient_data["readings"] = patient_data["readings"][:5]  # Limit to 5 readings
            patient_data["readings_shown"] = len(patient_data["readings"])
            distinct_patients.append(patient_data)
        
        # Sort by highest/lowest values
        if find_type == "high":
            distinct_patients.sort(key=lambda x: x["highest_value"], reverse=True)
        else:
            distinct_patients.sort(key=lambda x: x["lowest_value"])
        
        return distinct_patients
