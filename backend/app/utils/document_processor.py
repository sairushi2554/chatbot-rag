import os
import shutil
from typing import List
from pathlib import Path
from langchain.schema import Document
from ..config import settings
from ..services.vector_store import DocumentLoader, VectorStoreManager
from ..models.rag_model import RAGModel

def process_initial_documents():
    """Process documents from the documents directory and create/update the vector store."""
    try:
        logger.info(f"Starting document processing...")
        logger.info(f"Documents directory: {settings.DOCUMENTS_DIR}")
        logger.info(f"Vector store path: {settings.VECTOR_STORE_PATH}")
        
        # Create necessary directories
        os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
        os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
        
        # Check if documents exist
        if not os.path.exists(settings.DOCUMENTS_DIR):
            logger.error(f"Documents directory does not exist: {settings.DOCUMENTS_DIR}")
            return False
            
        files = list(Path(settings.DOCUMENTS_DIR).glob('*'))
        if not files:
            logger.warning(f"No files found in documents directory: {settings.DOCUMENTS_DIR}")
            return False
            
        logger.info(f"Found {len(files)} files in documents directory")
        
        # Initialize components
        rag_model = RAGModel()
        vector_store_manager = VectorStoreManager(rag_model.embeddings)
        
        # Process all documents in the documents directory
        all_documents = []
        supported_extensions = ['.pdf', '.txt', '.docx', '.md']
        processed_files = 0
        
        for file_path in files:
            if file_path.suffix.lower() not in supported_extensions:
                logger.warning(f"Skipping unsupported file: {file_path}")
                continue
                
            try:
                logger.info(f"Processing: {file_path}")
                documents = DocumentLoader.load_document(str(file_path))
                chunks = DocumentLoader.split_documents(
                    documents,
                    chunk_size=int(settings.CHUNK_SIZE),
                    chunk_overlap=int(settings.CHUNK_OVERLAP)
                )
                all_documents.extend(chunks)
                processed_files += 1
                logger.info(f"  - Processed {len(chunks)} chunks from {file_path.name}")
            except Exception as e:
                logger.error(f"  - Error processing {file_path}: {str(e)}", exc_info=True)
                continue
        
        if not all_documents:
            logger.error("No valid documents were processed.")
            return False
        
        # Create or update the vector store
        logger.info(f"Creating/updating vector store with {len(all_documents)} chunks from {processed_files} files...")
        vector_store_manager.create_vector_store(all_documents)
        vector_store_manager.save_vector_store(settings.VECTOR_STORE_PATH)
        
        logger.info(f"Vector store created/updated successfully at: {settings.VECTOR_STORE_PATH}")
        return True
        
    except Exception as e:
        logger.error(f"Error during document processing: {str(e)}", exc_info=True)
        return False

def check_vector_store() -> bool:
    """Check if a vector store exists and is valid."""
    if not os.path.exists(settings.VECTOR_STORE_PATH):
        return False
        
    try:
        rag_model = RAGModel()
        rag_model.load_vector_store(settings.VECTOR_STORE_PATH)
        return True
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        return False
