"""
Pinecone Utilities Module

This module provides reusable utilities for Pinecone vector database operations
to avoid code duplication across the application.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from pinecone import Vector

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class PineconeUtils:
    """Utility class for Pinecone operations"""
    
    _client = None
    _index = None
    _index_name: Optional[str] = None
    
    @classmethod
    def get_client(cls, api_key: Optional[str] = None):
        """
        Get or create Pinecone client instance (singleton pattern)
        
        Args:
            api_key: Optional API key. If not provided, uses environment variable
            
        Returns:
            Pinecone client instance or None if initialization fails
        """
        if cls._client is None:
            try:
                # Use provided API key or get from environment
                key = api_key or os.getenv("PINECONE_API_KEY")
                if not key:
                    logger.error("❌ Pinecone API key not found")
                    return None
                
                # Try new Pinecone API
                try:
                    from pinecone import Pinecone
                    cls._client = Pinecone(api_key=key)
                    logger.debug("✅ Pinecone client initialized successfully (new API)")
                except (ImportError, AttributeError) as e:
                    logger.error(f"❌ Failed to initialize Pinecone with new API: {e}")
                    return None
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize Pinecone client: {e}")
                return None
        
        return cls._client
    
    @classmethod
    def get_index(cls, index_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Get or create Pinecone index instance
        
        Args:
            index_name: Name of the index. If not provided, uses environment variable
            api_key: Optional API key for client initialization
            
        Returns:
            Pinecone index instance or None if initialization fails
        """
        # Use provided index name or get from environment
        target_index_name = index_name or os.getenv("INDEX_NAME", "hospital-handbook-index")
        
        # Check if we already have the correct index
        if cls._index is not None and cls._index_name == target_index_name:
            return cls._index
        
        try:
            # Get or create client
            client = cls.get_client(api_key)
            if not client:
                logger.error("❌ Cannot get Pinecone index without valid client")
                return None
            
            # List available indexes
            available_indexes = [idx.name for idx in client.list_indexes()]
            logger.debug(f"📋 Available Pinecone indexes: {available_indexes}")
            
            if target_index_name in available_indexes:
                cls._index = client.Index(target_index_name)
                cls._index_name = target_index_name
                logger.info(f"✅ Connected to Pinecone index: {target_index_name}")
                
                # Test index with describe_index_stats
                try:
                    stats = cls._index.describe_index_stats()
                    logger.debug(f"📊 Index stats: {stats}")
                except Exception as stats_error:
                    logger.warning(f"⚠️ Could not get index stats: {stats_error}")
                
                return cls._index
            else:
                logger.warning(f"❌ Pinecone index '{target_index_name}' not found in available indexes: {available_indexes}")
                logger.info(f"🔧 Creating new Pinecone index: {target_index_name}")
                
                # Create the index
                try:
                    from pinecone import ServerlessSpec
                    
                    # Get region from environment variable
                    region = os.getenv("PINECONE_REGION", "us-east-1")
                    logger.info(f"🌍 Using Pinecone region: {region}")
                    
                    client.create_index(
                        name=target_index_name,
                        dimension=1536,  # OpenAI embedding dimension
                        metric='cosine',
                        spec=ServerlessSpec(
                            cloud='aws',
                            region=region
                        )
                    )
                    
                    logger.info(f"✅ Created Pinecone index: {target_index_name}")
                    
                    # Wait for index to be ready and connect
                    import time
                    time.sleep(5)  # Wait for index to be ready
                    
                    cls._index = client.Index(target_index_name)
                    cls._index_name = target_index_name
                    
                    return cls._index
                    
                except Exception as create_error:
                    logger.error(f"❌ Failed to create Pinecone index: {create_error}")
                    return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get Pinecone index: {e}")
            logger.debug(f"🔍 Pinecone index error details: {type(e).__name__}: {str(e)}")
            return None
    
    @classmethod
    def reset_connection(cls):
        """Reset the client and index instances (useful for testing or config changes)"""
        cls._client = None
        cls._index = None
        cls._index_name = None
    
    @classmethod
    def query_index(
        cls,
        vector: List[float],
        top_k: int = 3,
        include_metadata: bool = True,
        index_name: Optional[str] = None,
        **kwargs
    ) :
        """
        Query the Pinecone index with a vector
        
        Args:
            vector: Query vector (embeddings)
            top_k: Number of top results to return
            include_metadata: Whether to include metadata in results
            index_name: Optional index name (uses default if not provided)
            **kwargs: Additional parameters for the query
            
        Returns:
            Query results dictionary or None if failed
        """
        try:
            index = cls.get_index(index_name)
            if not index:
                logger.error("❌ Pinecone index not available for querying")
                return None
            
            logger.debug(f"🔍 Querying Pinecone with vector of length {len(vector)}, top_k={top_k}")
            
            # Query the index
            results = index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=include_metadata,
                **kwargs
            )
            
            logger.debug(f"✅ Pinecone query successful, found {len(getattr(results, 'matches', []))} matches")
            return results
            
        except Exception as e:
            logger.error(f"❌ Pinecone query failed: {e}")
            return None
    
    @classmethod
    def upsert_vectors(
        cls,
        vectors: List[Vector],
        index_name: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Upsert vectors into the Pinecone index
        
        Args:
            vectors: List of vector dictionaries with id, values, and optional metadata
            index_name: Optional index name (uses default if not provided)
            **kwargs: Additional parameters for upsert
            
        Returns:
            True if successful, False otherwise
        """
        try:
            index = cls.get_index(index_name)
            if not index:
                logger.error("❌ Pinecone index not available for upserting")
                return False
            
            logger.debug(f"📤 Upserting {len(vectors)} vectors to Pinecone")
            
            # Upsert vectors
            index.upsert(vectors=vectors, **kwargs)
            
            logger.info(f"✅ Successfully upserted {len(vectors)} vectors to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pinecone upsert failed: {e}")
            return False
    
    @classmethod
    def delete_vectors(
        cls,
        ids: List[str],
        index_name: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Delete vectors from the Pinecone index
        
        Args:
            ids: List of vector IDs to delete
            index_name: Optional index name (uses default if not provided)
            **kwargs: Additional parameters for delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            index = cls.get_index(index_name)
            if not index:
                logger.error("❌ Pinecone index not available for deletion")
                return False
            
            logger.debug(f"🗑️ Deleting {len(ids)} vectors from Pinecone")
            
            # Delete vectors
            index.delete(ids=ids, **kwargs)
            
            logger.info(f"✅ Successfully deleted {len(ids)} vectors from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pinecone delete failed: {e}")
            return False
    
    @classmethod
    def get_index_stats(cls, index_name: Optional[str] = None):
        """
        Get statistics for the Pinecone index
        
        Args:
            index_name: Optional index name (uses default if not provided)
            
        Returns:
            Index statistics dictionary or None if failed
        """
        try:
            index = cls.get_index(index_name)
            if not index:
                logger.error("❌ Pinecone index not available for stats")
                return None
            
            stats = index.describe_index_stats()
            logger.debug(f"📊 Retrieved Pinecone index stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get Pinecone index stats: {e}")
            return None

# Convenience functions for quick usage
def get_pinecone_client(api_key: Optional[str] = None):
    """Get Pinecone client instance"""
    return PineconeUtils.get_client(api_key)

def get_pinecone_index(index_name: Optional[str] = None, api_key: Optional[str] = None):
    """Get Pinecone index instance"""
    return PineconeUtils.get_index(index_name, api_key)

def query_pinecone(
    vector: List[float],
    top_k: int = 3,
    include_metadata: bool = True,
    index_name: Optional[str] = None,
    **kwargs
):
    """
    Query Pinecone with a vector
    
    Args:
        vector: Query vector (embeddings)
        top_k: Number of top results to return
        include_metadata: Whether to include metadata
        index_name: Optional index name
        **kwargs: Additional parameters like filter
        
    Returns:
        Query results or None if failed
    """
    return PineconeUtils.query_index(
        vector=vector,
        top_k=top_k,
        include_metadata=include_metadata,
        index_name=index_name,
        **kwargs
    )

def upsert_to_pinecone(
    vectors: List[Vector],
    index_name: Optional[str] = None
) -> bool:
    """Upsert vectors to Pinecone"""
    return PineconeUtils.upsert_vectors(vectors, index_name)

def delete_from_pinecone(
    ids: List[str],
    index_name: Optional[str] = None
) -> bool:
    """Delete vectors from Pinecone"""
    return PineconeUtils.delete_vectors(ids, index_name)

def get_pinecone_stats(index_name: Optional[str] = None):
    """Get Pinecone index statistics"""
    return PineconeUtils.get_index_stats(index_name)

# Usage examples (for documentation):
"""
Example Usage:

1. Simple query:
   results = query_pinecone(
       vector=[0.1, 0.2, 0.3, ...],
       top_k=5
   )

2. Get index directly:
   index = get_pinecone_index("my-index")
   if index:
       # Use index for custom operations

3. Upsert vectors:
   vectors = [
       {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {"text": "example"}},
       {"id": "vec2", "values": [0.3, 0.4, ...], "metadata": {"text": "another"}}
   ]
   success = upsert_to_pinecone(vectors)

4. Using the class directly:
   results = PineconeUtils.query_index(
       vector=[0.1, 0.2, 0.3, ...],
       top_k=5,
       index_name="custom-index"
   )

5. Get index stats:
   stats = get_pinecone_stats()
   if stats:
       print(f"Total vectors: {stats.get('total_vector_count', 0)}")
"""
