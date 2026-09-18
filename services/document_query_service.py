"""
Document Query Service
Service for querying documents using RAG (Retrieval-Augmented Generation)
"""

import logging
from typing import Dict, Any, List, Optional

# Import utilities for document querying
try:
    from lib.openai_utils import create_embeddings, chat_completion
    from lib.pinecone_utils import query_pinecone
    QUERY_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    QUERY_DEPENDENCIES_AVAILABLE = False
    logging.error(f"Query dependencies not available: {e}")

logger = logging.getLogger(__name__)


class DocumentQueryService:
    """Service for querying documents using RAG"""
    
    def __init__(self):
        self.available = QUERY_DEPENDENCIES_AVAILABLE
    
    def query_documents(
        self,
        query: str,
        top_k: int = 5,
        max_tokens: int = 150,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Query documents using RAG (Retrieval-Augmented Generation)
        
        Args:
            query: The user's question
            top_k: Number of documents to retrieve
            max_tokens: Maximum tokens for response generation
            temperature: Temperature for response generation
            
        Returns:
            Dictionary with query results
        """
        try:
            if not self.available:
                return {
                    "success": False,
                    "query": query,
                    "response": "Document query service is not available",
                    "error": "Query dependencies not available",
                    "total_documents_found": 0
                }
            
            logger.info(f"🔍 Processing document query: {query}")
            
            # Step 1: Create embeddings for the query
            query_embedding = create_embeddings(query)
            if not query_embedding:
                return {
                    "success": False,
                    "query": query,
                    "response": "Failed to create embeddings for the query",
                    "error": "Embedding creation failed",
                    "total_documents_found": 0
                }
            
            # Normalize the embedding if it's a list of embeddings
            if isinstance(query_embedding, list) and len(query_embedding) > 0:
                if isinstance(query_embedding[0], list):
                    query_vector = query_embedding[0]  # Take first embedding
                else:
                    query_vector = query_embedding
            else:
                query_vector = query_embedding
            
            # Step 2: Query Pinecone for relevant documents
            results = query_pinecone(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            if not results:
                return {
                    "success": False,
                    "query": query,
                    "response": "I couldn't find relevant information in the knowledge base. Please try a different query.",
                    "total_documents_found": 0
                }
            
            # Step 3: Extract text from matches
            matches = results.get("matches", []) if isinstance(results, dict) else []
            if hasattr(results, "matches"):
                matches = getattr(results, "matches", [])
            
            if not matches:
                return {
                    "success": False,
                    "query": query,
                    "response": "I couldn't find relevant information in the knowledge base. Please try a different query.",
                    "total_documents_found": 0
                }
            
            # Extract context documents
            context_documents = []
            for match in matches:
                text = match.get("metadata", {}).get("text", "")
                if text:
                    context_documents.append(text)
            
            if not context_documents:
                return {
                    "success": False,
                    "query": query,
                    "response": "I couldn't find relevant information in the knowledge base. Please try a different query.",
                    "total_documents_found": 0
                }
            
            # Step 4: Generate response using OpenAI
            context = " ".join(context_documents)
            prompt = f"{context}\n\nUser query: {query}"
            
            response = chat_completion(
                prompt=prompt,
                system_message="You are a helpful medical assistant for Revival Hospital. Please answer the user's query based on the provided context. No need to say 'I am an AI model' or 'based on the document'. Do not repeat the question. Provide clear, helpful medical information.",
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if not response:
                return {
                    "success": False,
                    "query": query,
                    "response": "Sorry, I couldn't generate a response.",
                    "context_documents": context_documents,
                    "total_documents_found": len(context_documents)
                }
            
            return {
                "success": True,
                "query": query,
                "response": response.strip(),
                "context_documents": context_documents,
                "total_documents_found": len(context_documents)
            }
            
        except Exception as e:
            logger.error(f"❌ Document query failed: {e}")
            return {
                "success": False,
                "query": query,
                "response": f"Query failed: {str(e)}",
                "error": str(e),
                "total_documents_found": 0
            }
    
    def is_available(self) -> bool:
        """Check if the query service is available"""
        return self.available
    
    def get_status(self) -> Dict[str, Any]:
        """Get query service status"""
        return {
            "service": "Document Query Service",
            "available": self.available,
            "dependencies": {
                "openai_utils": QUERY_DEPENDENCIES_AVAILABLE,
                "pinecone_utils": QUERY_DEPENDENCIES_AVAILABLE
            }
        }


# Create a global instance
document_query_service = DocumentQueryService()
