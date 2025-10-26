# config.py

from pydantic_settings import BaseSettings
from enum import Enum
from pathlib import Path

# NOTE: No need for manual load_dotenv() when using BaseSettings' Config class.

class ModelProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    SENTENCE_TRANSFORMERS = "sentence-transformers"

class Settings(BaseSettings):
    # API settings
    APP_NAME: str = "RAG Chatbot API"
    DEBUG: bool = True
    VERSION: str = "1.0.0"

    # --- Provider Settings ---
    # Defines the primary provider for embeddings and LLMs.
    # Can be overridden by environment variables.
    EMBEDDING_PROVIDER: ModelProvider = ModelProvider.SENTENCE_TRANSFORMERS
    LLM_PROVIDER: ModelProvider = ModelProvider.OLLAMA

    # --- Model Names & Keys ---
    # Pydantic will automatically read these from environment variables if they exist.
    # Sentence-Transformers
    SENTENCE_TRANSFORMERS_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "llama2"
    OLLAMA_EMBEDDING_MODEL: str = "llama2" # Define the embedding model for Ollama

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    OPENAI_CHAT_MODEL: str = "gpt-3.5-turbo"

    # --- Paths ---
    # Use pathlib.Path for robust, object-oriented path management.
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DOCUMENTS_DIR: Path = BASE_DIR / 'documents'
    VECTOR_STORE_PATH: Path = BASE_DIR / 'vector_store'

    # --- Document Processing ---
    # Pydantic will correctly parse these as integers from the environment.
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # --- CORS settings ---
    ALLOWED_ORIGINS: list[str] = ["*"]

    # --- Dynamic Properties ---
    # These properties select the correct model based on the provider settings.
    @property
    def embedding_model_name(self) -> str:
        if self.EMBEDDING_PROVIDER == ModelProvider.OPENAI:
            return self.OPENAI_EMBEDDING_MODEL
        elif self.EMBEDDING_PROVIDER == ModelProvider.SENTENCE_TRANSFORMERS:
            return self.SENTENCE_TRANSFORMERS_EMBEDDING_MODEL
        elif self.EMBEDDING_PROVIDER == ModelProvider.OLLAMA:
            return self.OLLAMA_EMBEDDING_MODEL
        raise ValueError(f"Unsupported embedding provider: {self.EMBEDDING_PROVIDER}")

    @property
    def chat_model_name(self) -> str:
        if self.LLM_PROVIDER == ModelProvider.OLLAMA:
            return self.OLLAMA_CHAT_MODEL
        elif self.LLM_PROVIDER == ModelProvider.OPENAI:
            return self.OPENAI_CHAT_MODEL
        raise ValueError(f"Unsupported LLM provider: {self.LLM_PROVIDER}")

    class Config:
        # Pydantic will read from this file, making load_dotenv() redundant.
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        extra = 'ignore'

# Create a single, importable instance of the settings.
settings = Settings()