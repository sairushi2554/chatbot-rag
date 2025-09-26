from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import tempfile
import shutil
import logging
from pathlib import Path

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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models and services
rag_model = None
vector_store_manager = None
chat_service = None

# Initialize the RAG model and services
def initialize_services():
    global rag_model, vector_store_manager, chat_service
    
    try:
        rag_model = RAGModel()
        vector_store_manager = VectorStoreManager(rag_model.embeddings)
        
        # Load vector store if it exists
        if check_vector_store():
            rag_model.load_vector_store(settings.VECTOR_STORE_PATH)
            rag_model.initialize_qa_chain()
            chat_service = ChatService(rag_model.vector_store)
            logger.info("Vector store and chat service initialized successfully")
            return True
        else:
            logger.warning("No valid vector store found. Please process documents first.")
            return False
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        return False

# Initialize services on startup
initialize_services()

# Pydantic models for request/response validation
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

# API Endpoints
@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running"
    }

@app.post("/initialize/", status_code=status.HTTP_200_OK)
async def initialize_documents():
    """Initialize or update the knowledge base with documents from the documents directory."""
    try:
        success = process_initial_documents()
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to process documents. Please check the logs for more details."
            )
        
        # Reinitialize services with the new vector store
        if not initialize_services():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize services with the new vector store."
            )
            
        return {"message": "Documents processed and vector store updated successfully"}
    except Exception as e:
        logger.error(f"Error initializing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error initializing documents: {str(e)}"
        )

@app.post("/upload/", response_model=dict)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the documents directory for processing."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    # Ensure documents directory exists
    os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
    
    # Save the uploaded file
    file_path = os.path.join(settings.DOCUMENTS_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "message": "Document uploaded successfully",
            "file_path": str(file_path)
        }
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving uploaded file: {str(e)}"
        )

@app.post("/chat/", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """Handle chat messages with the RAG model."""
    if not chat_service or not rag_model.vector_store:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Knowledge base not initialized. Please process documents first."
        )
    
    try:
        # Process the message
        response = await chat_service.process_message(chat_request.message)
        return {
            "response": response["answer"],
            "sources": response.get("sources", [])
        }
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat message: {str(e)}"
        )

@app.post("/switch-model/")
async def switch_model(provider: str = Body(..., embed=True)):
    """Switch between OpenAI and Ollama providers."""
    try:
        if provider.lower() == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API key is not configured")
            settings.MODEL_PROVIDER = ModelProvider.OPENAI
        elif provider.lower() == "ollama":
            settings.MODEL_PROVIDER = ModelProvider.OLLAMA
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Reinitialize services with the new provider
        if not initialize_services():
            raise Exception("Failed to initialize services with the new model provider")
            
        return {
            "message": f"Switched to {provider} provider",
            "current_model": settings.chat_model
        }
    except Exception as e:
        logger.error(f"Error switching model provider: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error switching model provider: {str(e)}"
        )

@app.post("/reset/")
async def reset_chat():
    """Reset the chat history."""
    global chat_service
    if chat_service:
        chat_service.clear_history()
    return {"message": "Chat history cleared"}

# Check if vector store exists and initialize services
if check_vector_store():
    try:
        initialize_services()
    except Exception as e:
        logger.warning(f"Could not initialize services: {str(e)}")
else:
    logger.info("No existing vector store found. Please upload and process documents.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
