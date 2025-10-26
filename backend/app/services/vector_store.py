# vector store
import logging
from pathlib import Path
from typing import List, Dict, Type, Union, Optional

# --- LangChain Imports ---
# Use the community imports for better future-proofing and modularity
from langchain_core.documents import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings # Use core for base class
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)
from ..config import settings # Assuming this provides CHUNK_SIZE and CHUNK_OVERLAP

# --- Configure Logger ---
logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    Handles loading and splitting documents from various file formats.
    Optimized to use modern LangChain imports and handle common extensions.
    """

    # Map file extensions to their corresponding loader class
    # Use Path.suffix for key consistency
    LOADER_MAPPING: Dict[str, Type[BaseLoader]] = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
        ".md": UnstructuredMarkdownLoader,
    }

    @classmethod
    def load_document(cls, file_path: Union[str, Path]) -> List[Document]:
        """Loads a document from a file path using the appropriate loader."""
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            # Use logger for critical failure that should stop execution
            logger.error(f"File not found: {file_path_obj}")
            raise FileNotFoundError(f"File not found: {file_path_obj}")

        # Ensure the extension is consistent (lowercase)
        ext = file_path_obj.suffix.lower() 
        if ext not in cls.LOADER_MAPPING:
            logger.warning(f"Unsupported file type: {ext} for {file_path_obj.name}")
            raise ValueError(f"Unsupported file type: {ext}")

        # Instantiate and use the correct loader from the mapping
        loader_class = cls.LOADER_MAPPING[ext]
        # Path object is better than string for loader initialization
        loader = loader_class(str(file_path_obj)) 
        logger.info(f"Loading document: {file_path_obj.name} using {loader_class.__name__}")
        return loader.load()

    @staticmethod
    def split_documents(
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Document]:
        """Splits a list of documents into smaller chunks using configured or provided parameters."""
        
        # Use configured defaults if not provided, robustly accessing settings
        try:
            c_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
            c_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
        except AttributeError:
             logger.error("Configuration settings (CHUNK_SIZE/CHUNK_OVERLAP) are missing.")
             raise

        if not documents:
            logger.warning("Attempted to split an empty list of documents.")
            return []

        # Optimization: Default text splitter is RecursiveCharacterTextSplitter 
        # which is usually the best general-purpose choice.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            length_function=len, # Explicitly use 'len' (character count) for clarity
            add_start_index=True # Good for debugging/tracing source chunks
        )
        
        logger.info(f"Splitting {len(documents)} document(s) into chunks (size={c_size}, overlap={c_overlap}).")
        return text_splitter.split_documents(documents)


# ----------------------------------------------------------------------
class VectorStoreManager:
    """
    Manages all FAISS vector store operations (creation, saving, loading, and searching).
    """

    # Use a more generic type hint for the vector store
    def __init__(self, embeddings: Embeddings):
        """Initializes the manager with the embedding model."""
        self.embeddings = embeddings
        self.vector_store: Optional[FAISS] = None # Use Optional for clarity

    def create_vector_store(self, documents: List[Document]) -> None:
        """Creates a new vector store in memory from a list of documents."""
        if not documents:
             logger.warning("Attempted to create vector store from an empty document list.")
             return 
             
        logger.info(f"Creating new vector store from {len(documents)} document chunks...")
        # Use a try/except block to catch potential embedding/FAISS errors
        try:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info("Vector store creation complete.")
        except Exception as e:
            logger.error(f"Error during FAISS vector store creation: {e}")
            self.vector_store = None
            raise

    def save_vector_store(self, path: Union[str, Path]) -> None:
        """Saves the current in-memory vector store to disk."""
        if not self.vector_store:
            raise ValueError("Vector store is not initialized. Cannot save.")
        
        folder_path = Path(path)
        # Use exist_ok=True for atomic creation and avoiding race conditions
        folder_path.mkdir(parents=True, exist_ok=True) 
        
        logger.info(f"Saving vector store to: {folder_path.resolve()}") # Use .resolve() for full path
        self.vector_store.save_local(folder_path=str(folder_path))

    def load_vector_store(self, path: Union[str, Path]) -> None:
        """Loads a vector store from disk."""
        folder_path = Path(path)
        if not folder_path.exists():
            logger.error(f"Vector store directory not found at: {folder_path.resolve()}")
            raise FileNotFoundError(f"Vector store not found at: {folder_path}")

        logger.info(f"Loading vector store from: {folder_path.resolve()}")
        try:
            # The 'allow_dangerous_deserialization' flag is often required for older/complex stores.
            self.vector_store = FAISS.load_local(
                folder_path=str(folder_path),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True 
            )
            logger.info("Vector store loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load vector store from {folder_path.resolve()}. Error: {e}")
            self.vector_store = None
            raise

    def get_retriever(self, k: int = 4):
        """Returns a retriever for the initialized vector store."""
        if not self.vector_store:
            raise ValueError("Vector store is not initialized. Cannot create retriever.")
            
        # Optimization: Use the .as_retriever() method directly from the instance
        # It's an instance method, not a class method, so it's correct.
        logger.debug(f"Returning vector store retriever with k={k}.")
        return self.vector_store.as_retriever(search_kwargs={'k': k})