"""
Document Training Service for Revival Medical System
Handles document training, embedding creation, and Pinecone storage
"""

import os
import shutil
import logging
from typing import List, Dict, Any, Optional
import PyPDF2
from dotenv import load_dotenv
from lib.openai_utils import create_embeddings, normalize_embedding_vector
from lib.pinecone_utils import get_pinecone_index, upsert_to_pinecone

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class DocumentTrainingService:
    """Service for training hospital documents"""
    
    def __init__(self):
        self.index_name = os.getenv('INDEX_NAME')
        if not self.index_name:
            raise ValueError("INDEX_NAME not found in environment variables")
        
        self.documents_folder = 'documents/pdf'
        self.trained_folder = os.path.join(self.documents_folder, 'trained')
    
    def train_documents(self, folder_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train documents from the specified folder
        
        Args:
            folder_path: Optional custom folder path. If None, uses default documents/pdf
            
        Returns:
            Dictionary with training results
        """
        try:
            # Use custom folder path or default
            source_folder = folder_path or self.documents_folder
            
            logger.info(f"🚀 Starting document training process for folder: {source_folder}")
            
            # Initialize Pinecone and ensure index exists
            logger.info("🔧 Initializing Pinecone and checking index...")
            index = get_pinecone_index(self.index_name)
            if not index:
                return {
                    "success": False,
                    "error": "Failed to get or create Pinecone index",
                    "details": f"Index name: {self.index_name}"
                }
            
            logger.info(f"✅ Connected to Pinecone index: {self.index_name}")
            
            # Read and process PDFs
            documents = self._read_pdfs_from_folder(source_folder)
            
            if not documents:
                return {
                    "success": False,
                    "error": "No documents found to process",
                    "folder_path": source_folder
                }
            
            logger.info(f"📄 Total documents to process: {len(documents)}")
            
            # Store embeddings in Pinecone and move trained files
            successfully_processed = self._store_embeddings_in_pinecone(documents, source_folder)
            
            result = {
                "success": True,
                "message": "Documents training completed successfully!",
                "total_documents": len(documents),
                "successfully_processed": len(successfully_processed),
                "processed_files": successfully_processed,
                "index_name": self.index_name,
                "source_folder": source_folder,
                "trained_folder": os.path.join(source_folder, 'trained')
            }
            
            logger.info(f"✅ {result['message']}")
            logger.info(f"📊 Stored {len(documents)} document chunks in Pinecone index: {self.index_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Document training failed: {e}")
            return {
                "success": False,
                "error": f"Training failed: {str(e)}",
                "folder_path": folder_path or self.documents_folder
            }
    
    def _read_pdfs_from_folder(self, folder_path: str) -> List[Dict[str, str]]:
        """Read and process all PDF files from a folder"""
        documents = []
        
        if not os.path.exists(folder_path):
            logger.error(f"❌ Folder not found: {folder_path}")
            return documents
        
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
        logger.info(f"📚 Found {len(pdf_files)} PDF files in {folder_path}")
        
        for filename in pdf_files:
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    
                    if not text.strip():
                        logger.warning(f"⚠️  No text extracted from {filename}")
                        continue
                    
                    chunks = self._split_text_into_chunks(text)
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            'id': f"{filename}_chunk{i}",
                            'text': chunk,
                            'filename': filename,
                            'chunk_index': i
                        })
                    
                    logger.info(f"✅ Processed {filename}: {len(chunks)} chunks")
                    
            except Exception as e:
                logger.error(f"❌ Error reading {filename}: {e}")
        
        return documents
    
    def _split_text_into_chunks(self, text: str, max_tokens: int = 1000) -> List[str]:
        """Split text into smaller chunks for better embedding processing"""
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(" ".join(current_chunk)) > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
    
    def _store_embeddings_in_pinecone(self, documents: List[Dict[str, str]], source_folder: str) -> List[str]:
        """Store document embeddings in Pinecone using utility functions"""
        logger.info(f"📄 Processing {len(documents)} documents...")
        
        successfully_processed = []
        
        for i, doc in enumerate(documents):
            try:
                # Create embeddings using utility function
                embedding_result = create_embeddings(doc['text'])
                if not embedding_result:
                    logger.error(f"❌ Failed to create embedding for document {doc['id']}")
                    continue
                
                # Normalize embedding to single vector
                embedding = normalize_embedding_vector(embedding_result)
                if not embedding:
                    logger.error(f"❌ Failed to normalize embedding for document {doc['id']}")
                    continue

                # Prepare metadata for better search
                metadata = {
                    "text": doc['text'],
                    "filename": doc.get('filename', ''),
                    "chunk_index": doc.get('chunk_index', 0),
                    "document_type": "hospital_document",
                    "title": doc.get('filename', '').replace('.pdf', '').replace('_', ' ').title()
                }

                vec = {
                    'id': doc['id'],
                    'values': embedding,
                    'metadata': metadata
                }
                
                # Store in Pinecone using utility function
                success = upsert_to_pinecone(
                    index_name=self.index_name,
                    vectors=[vec]
                )
                
                if success:
                    logger.info(f"✅ Stored document {i+1}/{len(documents)}: {doc['id']}")
                    # Track the original filename for moving later
                    filename = doc['filename']
                    if filename and filename not in successfully_processed:
                        successfully_processed.append(filename)
                else:
                    logger.error(f"❌ Failed to store document {doc['id']}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing document {doc['id']}: {e}")
        
        # Move successfully processed files to trained folder
        if successfully_processed:
            self._move_trained_documents(source_folder, successfully_processed)
        
        return successfully_processed
    
    def _move_trained_documents(self, source_folder: str, processed_files: List[str]) -> None:
        """Move successfully processed documents to trained folder"""
        trained_folder = os.path.join(source_folder, 'trained')
        
        # Create trained folder if it doesn't exist
        if not os.path.exists(trained_folder):
            os.makedirs(trained_folder)
            logger.info(f"📁 Created trained folder: {trained_folder}")
        
        moved_count = 0
        for filename in processed_files:
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(trained_folder, filename)
            
            try:
                if os.path.exists(source_path):
                    # Check if file already exists in destination
                    if os.path.exists(destination_path):
                        logger.warning(f"⚠️  File already exists in trained folder: {filename}")
                        # Optionally remove the source file if it's identical
                        if os.path.getsize(source_path) == os.path.getsize(destination_path):
                            os.remove(source_path)
                            logger.info(f"🗑️  Removed duplicate source file: {filename}")
                    else:
                        shutil.move(source_path, destination_path)
                        moved_count += 1
                        logger.info(f"📦 Moved to trained: {filename}")
                else:
                    logger.warning(f"⚠️  Source file not found: {filename}")
                    
            except Exception as e:
                logger.error(f"❌ Error moving {filename}: {e}")
        
        if moved_count > 0:
            logger.info(f"✅ Successfully moved {moved_count} trained documents to {trained_folder}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and folder information"""
        try:
            # Check if folders exist
            source_exists = os.path.exists(self.documents_folder)
            trained_exists = os.path.exists(self.trained_folder)
            
            # Count files in each folder
            source_files = []
            trained_files = []
            
            if source_exists:
                source_files = [f for f in os.listdir(self.documents_folder) if f.endswith('.pdf')]
            
            if trained_exists:
                trained_files = [f for f in os.listdir(self.trained_folder) if f.endswith('.pdf')]
            
            return {
                "source_folder": self.documents_folder,
                "trained_folder": self.trained_folder,
                "source_folder_exists": source_exists,
                "trained_folder_exists": trained_exists,
                "source_files_count": len(source_files),
                "trained_files_count": len(trained_files),
                "source_files": source_files,
                "trained_files": trained_files,
                "index_name": self.index_name
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting training status: {e}")
            return {
                "error": f"Failed to get training status: {str(e)}",
                "source_folder": self.documents_folder,
                "trained_folder": self.trained_folder
            }

# Create global instance
training_service = DocumentTrainingService()
