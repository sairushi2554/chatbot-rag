from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import shutil
import logging
from pathlib import Path

# Local Imports
from .config import settings, ModelProvider
from .models.rag_model import RAGModel
from .services.vector_store import DocumentLoader, VectorStoreManager
from .services.chat_service import ChatService
from .utils.document_processor import process_initial_documents, check_vector_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="RAG-based Chatbot API"
)

# Configure CORS (omitted for brevity, assume it's correct)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG components
rag_model: Optional[RAGModel] = None
vector_store_manager: Optional[VectorStoreManager] = None
chat_service: Optional[ChatService] = None

# --- Core RAG Pipeline Setup Function (Replaces initialize_services) ---
def setup_rag_pipeline(force_reinitialize: bool = False) -> bool:
    """
    Initializes/reinitializes the RAG model, vector store, and chat service.
    If force_reinitialize is True, components are re-created even if a store exists.
    """
    global rag_model, vector_store_manager, chat_service
    
    # 1. Initialize core components (RAGModel provides Embeddings)
    try:
        # NOTE: RAGModel initializes the heavy HuggingFaceEmbeddings model
        rag_model = RAGModel()
        vector_store_manager = VectorStoreManager(rag_model.embeddings)
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize RAGModel/Embeddings: {str(e)}")
        return False

    # 2. Check and load vector store
    if check_vector_store() and not force_reinitialize:
        try:
            rag_model.load_vector_store(settings.VECTOR_STORE_PATH)
            rag_model.initialize_qa_chain()
            chat_service = ChatService(rag_model.vector_store)
            logger.info("Vector store and chat service loaded and initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading existing vector store: {str(e)}. Will attempt to process documents.")
            # Fall through to the processing step below if loading fails
    
    # 3. Process documents and initialize (either if no store exists or if loading failed)
    logger.info("Starting initial document processing from scratch...")
    success = process_initial_documents()
    
    if success:
        try:
            # Load the newly created store
            rag_model.load_vector_store(settings.VECTOR_STORE_PATH)
            rag_model.initialize_qa_chain()
            chat_service = ChatService(rag_model.vector_store)
            logger.info("Documents processed and RAG pipeline fully set up.")
            return True
        except Exception as e:
            logger.error(f"Error initializing services with the newly created vector store: {str(e)}")
            return False
    else:
        logger.warning("No documents were processed and no valid vector store could be created/loaded.")
        return False

# Pydantic models (omitted for brevity)
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []

class UploadResponse(BaseModel):
    message: str
    document_id: str
    num_chunks: int

# --- API Endpoints ---

@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running"
    }

@app.post("/initialize/", status_code=status.HTTP_200_OK)
async def initialize_documents():
    """Manually reinitialize the entire knowledge base by re-processing all documents."""
    try:
        # Force re-initialization (re-process all documents)
        if setup_rag_pipeline(force_reinitialize=True):
            return {"message": "Documents re-processed and vector store updated successfully."}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to process documents. Please check the documents directory and logs."
            )
    except Exception as e:
        logger.error(f"Error initializing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error initializing documents: {str(e)}"
        )

# ... (upload_document, chat, switch_model, reset_chat endpoints remain the same, 
# but they now call setup_rag_pipeline internally if needed or rely on the global setup) ...

# NOTE: The switch_model endpoint must be updated to call the new setup function:
@app.post("/switch-model/")
async def switch_model(provider: str = Body(..., embed=True)):
    """Switch between model providers and reinitialize services."""
    # ... (ModelProvider settings update logic remains the same) ...
    
    # Reinitialize services with the new provider
    if not setup_rag_pipeline(): # Don't force re-processing, just reload store with new embeddings/LLM
        raise Exception("Failed to set up RAG pipeline with the new model provider")
            
    # ... (Return success message) ...


# --- Application Startup Logic (Auto-Processing Implemented Here) ---
@app.on_event("startup")
def startup_event():
    """Run the main setup logic when the application starts."""
    logger.info("Starting application setup...")
    setup_rag_pipeline()
    logger.info("Application setup complete.")

# NOTE: The old logic outside of a function has been replaced by the @app.on_event("startup") handler.
# The original lines below must be removed:
# if check_vector_store():
#     try:
#         initialize_services() # REMOVE
#     except Exception as e:
#         logger.warning(f"Could not initialize services: {str(e)}")
# else:
#     logger.info("No existing vector store found. Please upload and process documents.") # REMOVE
# initialize_services() # REMOVE

if __name__ == "__main__":
    import uvicorn
    # Use the app instance directly
    uvicorn.run(app, host="0.0.0.0", port=8800, reload=True)