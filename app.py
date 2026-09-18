#!/usr/bin/env python3
"""
Revival Medical System FastAPI Application
Hospital chatbot API with LangChain agent and conversation memory
"""

import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key from environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# LangChain imports
try:
    from langchain.agents import create_openai_tools_agent, AgentExecutor
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.schema import HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"LangChain not available: {e}")
    print("Install with: pip install langchain langchain-openai")

# Import medical system components
try:
    from dal.database import init_database
    MCP_AVAILABLE = True
except ImportError as e:
    init_database = None
    MCP_AVAILABLE = False
    print(f"Medical system not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🔧 Starting Revival Medical System initialization...")
    
    try:
        # Check OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            logger.info("✅ OpenAI API key found")
        else:
            logger.warning("❌ OpenAI API key not found")
        
        # Initialize database
        if init_database:
            try:
                init_database()
                logger.info("✅ Medical database initialized successfully")
            except Exception as db_error:
                logger.error(f"❌ Database initialization failed: {db_error}")
        else:
            logger.warning("⚠️ Database initialization not available")

        # CGM module: verify/create its tables (non-fatal if DB not reachable)
        try:
            from database import create_tables as cgm_create_tables
            cgm_create_tables()
            logger.info("✅ CGM tables verified/created")
        except Exception as e:
            logger.warning(f"⚠️ CGM table check failed (non-fatal): {e}")
        
        # Summary
        logger.info("🚀 Revival Medical System API started successfully!")
        logger.info(f"📊 Component Status:")
        logger.info(f"   - OpenAI: {'✅' if openai_api_key else '❌'}")
        logger.info(f"   - Database: {'✅' if init_database else '❌'}")
        logger.info(f"   - LangChain: {'✅' if LANGCHAIN_AVAILABLE else '❌'}")
        logger.info(f"   - MCP: {'✅' if MCP_AVAILABLE else '❌'}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize medical system: {e}")
    
    yield
    
    logger.info("🛑 Revival Medical System API shutdown complete")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Revival Medical System API",
    description="Hospital chatbot API with LangChain agent and conversation memory",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
#
# allow_credentials=False is intentional and correct here.
#
# This app authenticates via Bearer token in the Authorization header and
# X-User-ID header — NOT via cookies or HTTP auth. Bearer tokens are just
# regular request headers; they do not require credentials mode.
#
# allow_credentials=True conflicts with allow_origins=["*"]:
# the browser security spec forbids wildcards when credentials=True,
# so the browser silently drops the response -> frontend gets no data
# and auth calls return 401 even though the server responded correctly.
#
# Setting credentials=False + origins=["*"] is the correct combination
# for a token-based API that needs to be callable from any frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware (from CGM module)
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.time() - start)*1000:.1f}ms"
    return response

# Global exception handler (from CGM module)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )

# Import and include API routes
try:
    from api.auth_routes import router as auth_router
    app.include_router(auth_router)
    from api.chat_routes import router as chat_router
    app.include_router(chat_router)
    from api.document_routes import router as document_router
    app.include_router(document_router)
    from api.metrics_routes import router as metrics_router  
    app.include_router(metrics_router) 
    from prediction import router as cgm_router
    app.include_router(cgm_router)
    from api.clinical_routes import router as clinical_router
    app.include_router(clinical_router)
    logger.info("✅ API routes loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to load API routes: {e}")
except Exception as e:
    logger.error(f"❌ Error including API routes: {e}")


# Health check endpoint (from CGM module)
@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "app": "Revival Medical System API",
        "version": "1.0.0",
    }


# Root endpoint
@app.get("/", tags=["System"])
def root():
    return {
        "message": "Welcome to Revival Medical System API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }