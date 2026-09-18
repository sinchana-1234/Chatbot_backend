"""
OpenAI Utilities Module

This module provides reusable utilities for OpenAI client creation and chat completions
to avoid code duplication across the application.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class OpenAIUtils:
    """Utility class for OpenAI operations"""
    
    _client: Optional[OpenAI] = None
    
    @classmethod
    def get_client(cls, api_key: Optional[str] = None) -> Optional[OpenAI]:
        """
        Get or create OpenAI client instance (singleton pattern)
        
        Args:
            api_key: Optional API key. If not provided, uses environment variable
            
        Returns:
            OpenAI client instance or None if initialization fails
        """
        if cls._client is None:
            try:
                # Use provided API key or get from environment
                key = api_key or os.getenv("OPENAI_API_KEY")
                if not key:
                    logger.error("❌ OpenAI API key not found")
                    return None
                
                cls._client = OpenAI(api_key=key)
                logger.debug("✅ OpenAI client initialized successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI client: {e}")
                return None
        
        return cls._client
    
    @classmethod
    def reset_client(cls):
        """Reset the client instance (useful for testing or key changes)"""
        cls._client = None
    
    @classmethod
    def create_chat_completion(
        cls,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 800,
        temperature: float = 0.3,
        system_message: Optional[str] = None,
        user_message: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Create a chat completion with common parameters
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: OpenAI model to use
            max_tokens: Maximum tokens in response
            temperature: Response randomness (0.0 to 2.0)
            system_message: Optional system message (will be prepended)
            user_message: Optional user message (will be appended)
            **kwargs: Additional parameters for the API call
            
        Returns:
            Response content as string or None if failed
        """
        try:
            client = cls.get_client()
            if not client:
                logger.error("❌ OpenAI client not available")
                return None
            
            # Build messages list
            final_messages = []
            
            # Add system message if provided
            if system_message:
                final_messages.append({"role": "system", "content": system_message})
            
            # Add provided messages
            final_messages.extend(messages)
            
            # Add user message if provided
            if user_message:
                final_messages.append({"role": "user", "content": user_message})
            
            logger.debug(f"🤖 Creating chat completion with {len(final_messages)} messages")
            
            # Create completion
            response = client.chat.completions.create(
                model=model,
                messages=final_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            content = (response.choices[0].message.content or "").strip()
            logger.debug(f"✅ Chat completion successful: {len(content)} characters")
            
            return content
            
        except Exception as e:
            logger.error(f"❌ Chat completion failed: {e}")
            return None
    
    @classmethod
    def create_embeddings(
        cls,
        text: Union[str, List[str]],
        model: str = "text-embedding-ada-002"
    ) :
        """
        Create embeddings for text
        
        Args:
            text: Text string or list of texts to embed
            model: Embedding model to use
            
        Returns:
            List of embedding values or None if failed
        """
        try:
            client = cls.get_client()
            if not client:
                logger.error("❌ OpenAI client not available for embeddings")
                return None
            
            response = client.embeddings.create(
                model=model,
                input=text
            )
            
            # Return first embedding if single text, otherwise return all
            if isinstance(text, str):
                return response.data[0].embedding
            else:
                return [item.embedding for item in response.data]
                
        except Exception as e:
            logger.error(f"❌ Embedding creation failed: {e}")
            return None

# Convenience functions for quick usage
def get_openai_client(api_key: Optional[str] = None) -> Optional[OpenAI]:
    """Get OpenAI client instance"""
    return OpenAIUtils.get_client(api_key)

def chat_completion(
    prompt: str,
    system_message: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 800,
    temperature: float = 0.3,
    **kwargs
) -> Optional[str]:
    """
    Simple chat completion with a single prompt
    
    Args:
        prompt: User prompt/question
        system_message: Optional system instruction
        model: OpenAI model to use
        max_tokens: Maximum tokens in response
        temperature: Response randomness
        **kwargs: Additional parameters
        
    Returns:
        Response content or None if failed
    """
    messages = [{"role": "user", "content": prompt}]
    
    return OpenAIUtils.create_chat_completion(
        messages=messages,
        system_message=system_message,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )

def advanced_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 800,
    temperature: float = 0.3,
    **kwargs
) -> Optional[str]:
    """
    Advanced chat completion with multiple messages
    
    Args:
        messages: List of message dictionaries
        model: OpenAI model to use
        max_tokens: Maximum tokens in response
        temperature: Response randomness
        **kwargs: Additional parameters
        
    Returns:
        Response content or None if failed
    """
    return OpenAIUtils.create_chat_completion(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )

from typing import Union

def normalize_embedding_vector(
    embedding_result: Optional[Union[List[float], List[List[float]]]]
) -> Optional[List[float]]:
    """
    Normalize embedding result to a single vector (List[float])
    
    Args:
        embedding_result: Result from create_embeddings function
        
    Returns:
        Single embedding vector as List[float] or None if invalid
    """
    if not embedding_result:
        return None
    
    if isinstance(embedding_result, list):
        # If it's a list of lists (multiple embeddings), take the first one
        if len(embedding_result) > 0 and isinstance(embedding_result[0], list):
            return embedding_result[0]  # type: ignore - we know this is List[float]
        else:
            # It's already a single vector - ensure it's List[float]
            return embedding_result  # type: ignore - we know this is List[float]
    
    return None

def create_embeddings(
    text: Union[str, List[str]], 
    model: str = "text-embedding-ada-002"
) -> Optional[Union[List[float], List[List[float]]]]:
    """Create embeddings for text"""
    return OpenAIUtils.create_embeddings(text, model)

# Predefined system messages for common use cases
SYSTEM_MESSAGES = {
    "school_assistant": "You are a helpful school assistant with access to school policies and student data. Provide clear, accurate, and helpful responses.",
    
    "policy_expert": "You are a knowledgeable school administrator who explains policies clearly and thoroughly. Use bullet points and structured formatting when appropriate.",
    
    "data_formatter": "You are a helpful assistant who formats technical data into user-friendly responses. Present information clearly and remove technical jargon.",
    
    "rag_assistant": "You are a helpful assistant that answers questions based on provided context. If the context doesn't contain relevant information, say so politely and suggest asking more specific questions.",
    
    "student_data_helper": "You are a school data assistant who helps format and present student information in a clear, privacy-conscious manner."
}

# Usage examples (for documentation):
"""
Example Usage:

1. Simple chat completion:
   response = chat_completion(
       prompt="What is the school's leave policy?",
       system_message=SYSTEM_MESSAGES["policy_expert"]
   )

2. Advanced chat with conversation history:
   messages = [
       {"role": "user", "content": "Tell me about leave policies"},
       {"role": "assistant", "content": "Here are the leave policies..."},
       {"role": "user", "content": "What about maternity leave?"}
   ]
   response = advanced_chat_completion(messages)

3. Get client for custom operations:
   client = get_openai_client()
   if client:
       # Use client for custom API calls

4. Create embeddings:
   embeddings = create_embeddings("This is some text to embed")

5. Using the class directly:
   response = OpenAIUtils.create_chat_completion(
       messages=[{"role": "user", "content": "Hello"}],
       system_message="You are a helpful assistant",
       temperature=0.5
   )
"""
