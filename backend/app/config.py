from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from enum import Enum
import os
from typing import Literal

# Load environment variables from .env file
load_dotenv()

class ModelProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    SENTENCE_TRANSFORMERS = "sentence-transformers"

class Settings(BaseSettings):
    # API settings
    APP_NAME: str = "RAG Chatbot API"
    DEBUG: bool = True
    VERSION: str = "1.0.0"
    
    # Model provider (sentence-transformers for embeddings, ollama for LLM)
    MODEL_PROVIDER: ModelProvider = ModelProvider.SENTENCE_TRANSFORMERS
    
    # Embedding Model (MiniLM)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Ollama LLM Settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_CHAT_MODEL: str = os.getenv("LLM_MODEL", "llama2")
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DOCUMENTS_DIR: str = os.path.join(BASE_DIR, 'documents')
    VECTOR_STORE_PATH: str = os.path.join(BASE_DIR, 'vector_store')
    
    # OpenAI settings (kept for reference, not used in this setup)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    OPENAI_CHAT_MODEL: str = "gpt-3.5-turbo"
    
    # Document processing settings
    CHUNK_SIZE: int = os.getenv("CHUNK_SIZE", 1000)
    CHUNK_OVERLAP: int = os.getenv("CHUNK_OVERLAP", 200)
    
    # CORS settings
    ALLOWED_ORIGINS: list = ["*"]
    
    @property
    def embedding_model(self) -> str:
        if self.MODEL_PROVIDER == ModelProvider.OPENAI:
            return self.OPENAI_EMBEDDING_MODEL
        elif self.MODEL_PROVIDER == ModelProvider.SENTENCE_TRANSFORMERS:
            return self.EMBEDDING_MODEL
        else:  # OLLAMA
            return self.OLLAMA_EMBEDDING_MODEL
    
    @property
    def chat_model(self) -> str:
        return self.OLLAMA_CHAT_MODEL if self.MODEL_PROVIDER != ModelProvider.OPENAI else self.OPENAI_CHAT_MODEL
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        extra = 'ignore'  # Ignore extra environment variables

# Create settings instance
settings = Settings()
