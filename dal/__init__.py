"""
Initialization file for the MCP system package
"""

from .database import get_db_manager, DatabaseManager, init_database

__version__ = "1.0.0"
__all__ = ["get_db_manager", "DatabaseManager", "init_database"]