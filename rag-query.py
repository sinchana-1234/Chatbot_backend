import os
from dotenv import load_dotenv
from lib.openai_utils import  create_embeddings, chat_completion, SYSTEM_MESSAGES, normalize_embedding_vector
from lib.pinecone_utils import  query_pinecone, get_pinecone_index

# Load environment variables from .env file
load_dotenv()

# Initialize utilities (clients will be created on first use)
print("Initializing OpenAI and Pinecone utilities...")

# Get index name from environment
index_name = os.getenv('INDEX_NAME')
if not index_name:
    print("❌ INDEX_NAME not found in environment variables")
    exit(1)

# Function to retrieve documents from Pinecone
def retrieve_documents_from_pinecone(query, top_k=5):
    """Retrieve documents from Pinecone using the utility function"""
    try:
        # Create embeddings using utility function
        query_embedding = create_embeddings(query)
        if not query_embedding:
            print("❌ Failed to create embeddings for query")
            return []
        
        # Normalize to single vector using utility function
        embedding_vector = normalize_embedding_vector(query_embedding)
        if not embedding_vector:
            print("❌ Invalid embedding format")
            return []
        
        # Query Pinecone using utility function
        results = query_pinecone(
            vector=embedding_vector,  # type: ignore - we've validated this is List[float]
            top_k=top_k,
            include_metadata=True,
            index_name=index_name
        )
        
        if not results or not isinstance(results, dict) or 'matches' not in results:
            print("❌ No results returned from Pinecone")
            return []
            
        return [match['metadata']['text'] for match in results['matches'] if 'metadata' in match and 'text' in match['metadata']]
        
    except Exception as e:
        print(f"❌ Error retrieving documents from Pinecone: {e}")
        return []

# Function to generate response using retrieved context
def generate_response(context, user_query):
    """Generate response using OpenAI chat completion utility"""
    try:
        prompt = f"{context}\n\nUser query: {user_query}"
        
        # Use chat completion utility with predefined system message
        response = chat_completion(
            prompt=prompt,
            system_message=SYSTEM_MESSAGES["rag_assistant"],
            model="gpt-3.5-turbo",
            max_tokens=150
        )
        
        if response:
            return response.strip()
        else:
            return "❌ No response generated"
            
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "❌ Failed to generate response"


while True:
    user_query = input("Enter your query (type 'exit' to quit): ")
    if user_query.lower() == "exit":
        print("Exiting...")
        break
    context = " ".join(retrieve_documents_from_pinecone(user_query))
    response = generate_response(context, user_query)
    print("Response:", response)