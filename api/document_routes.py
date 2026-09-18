"""
Document Training API Routes
API endpoints for training hospital documents and creating embeddings
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from services import training_service, document_query_service


logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/document", tags=["document"])

# Request/Response models
class TrainingRequest(BaseModel):
    folder_path: Optional[str] = None

class TrainingResponse(BaseModel):
    success: bool
    message: str
    total_documents: Optional[int] = None
    successfully_processed: Optional[int] = None
    processed_files: Optional[list] = None
    index_name: Optional[str] = None
    folder_path: Optional[str] = None
    error: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.3

class QueryResponse(BaseModel):
    success: bool
    query: str
    response: str
    context_documents: Optional[List[str]] = None
    total_documents_found: Optional[int] = None
    error: Optional[str] = None

@router.post("/train", response_model=TrainingResponse)
async def train_documents(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train documents and create embeddings (async)"""
    try:
        
        logger.info(f"📚 Starting document training for folder: {request.folder_path or 'default'}")
        
        # Run training in background
        def run_training():
            try:
                result = training_service.train_documents(request.folder_path)
                logger.info(f"✅ Training completed: {result}")
            except Exception as e:
                logger.error(f"❌ Background training failed: {e}")
        
        background_tasks.add_task(run_training)
        
        return TrainingResponse(
            success=True,
            message="Document training started in background. Check training status endpoint for progress.",
            folder_path=request.folder_path or "documents/pdf"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Training request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/train/sync", response_model=TrainingResponse)
async def train_documents_sync(request: TrainingRequest):
    """Train documents synchronously and return results"""
    try:
        
        logger.info(f"📚 Starting synchronous document training for folder: {request.folder_path or 'default'}")
        
        # Run training synchronously
        result = training_service.train_documents(request.folder_path)
        
        if result.get("success"):
            return TrainingResponse(
                success=True,
                message=result.get("message", "Training completed"),
                total_documents=result.get("total_documents"),
                successfully_processed=result.get("successfully_processed"),
                processed_files=result.get("processed_files"),
                index_name=result.get("index_name"),
                folder_path=request.folder_path
            )
        else:
            return TrainingResponse(
                success=False,
                message="Training failed",
                error=result.get("error"),
                folder_path=request.folder_path
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Synchronous training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.get("/status")
async def get_training_status():
    """Get current document training status"""
    try:
        
        status = training_service.get_training_status()
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using RAG (Retrieval-Augmented Generation)"""
    try:        
        # Use the document query service
        result = document_query_service.query_documents(
            query=request.query,
            top_k=request.top_k or 5,
            max_tokens=request.max_tokens or 150,
            temperature=request.temperature or 0.3
        )
        
        # Convert service result to API response
        if result.get("success"):
            return QueryResponse(
                success=True,
                query=result["query"],
                response=result["response"],
                context_documents=result.get("context_documents"),
                total_documents_found=result.get("total_documents_found")
            )
        else:
            return QueryResponse(
                success=False,
                query=result["query"],
                response=result["response"],
                context_documents=result.get("context_documents"),
                total_documents_found=result.get("total_documents_found", 0),
                error=result.get("error")
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Document query endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.get("/")
async def document_info():
    """Get document service information"""
    return {
        "service": "Document Training API",
        "version": "1.0.0",
        "endpoints": [
            "/api/document/train",
            "/api/document/train/sync", 
            "/api/document/status",
            "/api/document/query"
        ],
        "description": "API for training hospital documents, creating embeddings, and querying documents using RAG"
    }
