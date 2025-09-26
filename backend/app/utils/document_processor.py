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
    # Create necessary directories
    os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
    
    # Initialize components
    rag_model = RAGModel()
    vector_store_manager = VectorStoreManager(rag_model.embeddings)
    
    # Check if documents exist
    if not os.path.exists(settings.DOCUMENTS_DIR) or not os.listdir(settings.DOCUMENTS_DIR):
        print(f"No documents found in {settings.DOCUMENTS_DIR}. Please add documents and try again.")
        return False
    
    try:
        # Process all documents in the documents directory
        all_documents = []
        supported_extensions = ['.pdf', '.txt', '.docx', '.md']
        
        for file_path in Path(settings.DOCUMENTS_DIR).rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    print(f"Processing: {file_path}")
                    documents = DocumentLoader.load_document(str(file_path))
                    chunks = DocumentLoader.split_documents(
                        documents,
                        chunk_size=settings.CHUNK_SIZE,
                        chunk_overlap=settings.CHUNK_OVERLAP
                    )
                    all_documents.extend(chunks)
                    print(f"  - Processed {len(chunks)} chunks from {file_path.name}")
                except Exception as e:
                    print(f"  - Error processing {file_path}: {str(e)}")
        
        if not all_documents:
            print("No valid documents found to process.")
            return False
        
        # Create or update the vector store
        print(f"\nCreating/updating vector store with {len(all_documents)} chunks...")
        vector_store_manager.create_vector_store(all_documents)
        vector_store_manager.save_vector_store(settings.VECTOR_STORE_PATH)
        
        print(f"\nVector store created/updated successfully at: {settings.VECTOR_STORE_PATH}")
        return True
        
    except Exception as e:
        print(f"Error during document processing: {str(e)}")
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
