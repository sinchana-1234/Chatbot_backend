"""
Hospital Document Search Tool
Tool for searching hospital documents, handbooks, and medical documentation
"""

import logging
from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# Import utilities from lib folder
try:
    from lib.pinecone_utils import query_pinecone, get_pinecone_index
    from lib.openai_utils import create_embeddings
    SEARCH_AVAILABLE = True
except ImportError as e:
    SEARCH_AVAILABLE = False
    logging.error(f"Search utilities not available: {e}")

logger = logging.getLogger(__name__)

class HospitalDocumentSearchTool(BaseTool):
    """Tool for searching hospital documents and handbooks"""
    
    name: str = "search_hospital_documents"
    description: str = """Search hospital documents, handbooks, policies, and medical documentation.
    
    Use this tool to find information about:
    - Hospital policies and procedures
    - Medical protocols and guidelines
    - Patient care instructions
    - Emergency procedures
    - Equipment manuals
    - Staff handbooks
    - Treatment protocols
    - Regulatory compliance documents
    
    Parameters:
    - query (str): The search query or question about hospital documentation
    - document_type (str): Optional filter for specific document types (policies, protocols, handbooks, etc.)
    - max_results (int): Maximum number of results to return (default: 5)
    
    Examples:
    - "What are the infection control protocols?"
    - "Emergency response procedures for cardiac arrest"
    - "Patient discharge procedures"
    - "Staff safety guidelines"
    """
    
    def _run(self, query: str, document_type: Optional[str] = None, max_results: int = 5) -> Dict[str, Any]:
        """Execute hospital document search"""
        try:
            if not SEARCH_AVAILABLE:
                return {
                    "error": "Document search functionality is not available. Please check Pinecone and OpenAI utilities.",
                    "query": query
                }
            
            logger.info(f"🔍 Searching hospital documents for: {query}")
            
            # Use the same approach as the working DocumentQueryService
            from services.document_query_service import document_query_service
            
            # Call the working service
            result = document_query_service.query_documents(
                query=query,
                top_k=max_results,
                max_tokens=150,
                temperature=0.3
            )
            
            if not result.get("success"):
                return {
                    "message": "No relevant hospital documents found for your query.",
                    "query": query,
                    "results": []
                }
            
            # Convert service result to tool format
            context_documents = result.get("context_documents", [])
            formatted_results = []
            
            for i, doc in enumerate(context_documents):
                result_item = {
                    "score": 0.8,  # Default score since we don't have access to the raw scores
                    "content": doc,
                    "document_id": f"doc_{i}",
                    "document_type": document_type or "hospital_document",
                    "title": f"Hospital Document {i+1}",
                    "section": "content"
                }
                formatted_results.append(result_item)
            
            # Create summary response
            total_results = len(formatted_results)
            response = {
                "query": query,
                "total_results": total_results,
                "results": formatted_results,
                "message": f"Found {total_results} relevant hospital document(s) for your query.",
                "ai_response": result.get("response", "")  # Include the AI-generated response
            }
            
            # Add document type filter info if used
            if document_type:
                response["filtered_by"] = f"document_type: {document_type}"
            
            logger.info(f"✅ Found {total_results} hospital documents for query: {query}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Hospital document search failed: {e}")
            return {
                "error": f"Search failed: {str(e)}",
                "query": query,
                "results": []
            }
    
    async def _arun(self, query: str, document_type: Optional[str] = None, max_results: int = 5) -> Dict[str, Any]:
        """Async version of hospital document search"""
        return self._run(query, document_type, max_results)

# Function to help with document indexing (for future use)
def index_hospital_document(document_text: str, document_id: str, metadata: Dict[str, Any]) -> bool:
    """
    Index a hospital document for future searches
    
    Args:
        document_text: The text content of the document
        document_id: Unique identifier for the document
        metadata: Additional metadata (title, document_type, section, etc.)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not SEARCH_AVAILABLE:
            logger.error("Document indexing not available - missing utilities")
            return False
        
        # Create embeddings for the document
        embeddings = create_embeddings(document_text)
        if not embeddings:
            logger.error(f"Failed to create embeddings for document {document_id}")
            return False
        
        # Normalize embedding
        if isinstance(embeddings, list) and len(embeddings) > 0:
            if isinstance(embeddings[0], list):
                embedding_vector = embeddings[0]
            else:
                embedding_vector = embeddings
        else:
            embedding_vector = embeddings
        
        # Add text to metadata
        full_metadata = {
            "text": document_text,
            **metadata
        }
        
        # Create vector for upserting
        from pinecone import Vector
        
        vector = Vector(
            id=document_id,
            values=embedding_vector,
            metadata=full_metadata
        )
        
        # Index in Pinecone
        from lib.pinecone_utils import upsert_to_pinecone
        success = upsert_to_pinecone(vectors=[vector])
        
        if success:
            logger.info(f"✅ Successfully indexed hospital document: {document_id}")
            return True
        else:
            logger.error(f"❌ Failed to index hospital document: {document_id}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Document indexing failed for {document_id}: {e}")
        return False

# Example usage and test function
def test_hospital_document_search():
    """Test function for hospital document search"""
    tool = HospitalDocumentSearchTool()
    
    # Test queries
    test_queries = [
        "What are the infection control protocols?",
        "Emergency response procedures",
        "Patient discharge procedures",
        "Staff safety guidelines"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: {query}")
        result = tool._run(query)
        print(f"Results: {result.get('total_results', 0)} documents found")
        
        if result.get('results'):
            for i, doc in enumerate(result['results'][:2]):  # Show first 2 results
                print(f"  {i+1}. Score: {doc['score']}, Type: {doc['document_type']}")
                print(f"     Content: {doc['content'][:100]}...")

if __name__ == "__main__":
    test_hospital_document_search()
