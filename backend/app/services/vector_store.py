from typing import List, Optional
import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from ..config import settings

class DocumentLoader:
    """Handles loading documents from various file formats."""
    
    @staticmethod
    def load_document(file_path: str) -> List[Document]:
        """Load a document from a file path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
        elif file_extension in ['.doc', '.docx']:
            loader = Docx2txtLoader(file_path)
        elif file_extension == '.md':
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
            
        return loader.load()
    
    @staticmethod
    def split_documents(documents: List[Document], 
                       chunk_size: int = None, 
                       chunk_overlap: int = None) -> List[Document]:
        """Split documents into chunks."""
        chunk_size = chunk_size or settings.CHUNK_SIZE
        chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        return text_splitter.split_documents(documents)


class VectorStoreManager:
    """Manages the vector store operations."""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vector_store = None
        
    def create_vector_store(self, documents: List[Document]) -> None:
        """Create a new vector store from documents."""
        from langchain.vectorstores import FAISS
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
    def load_vector_store(self, path: str) -> None:
        """Load an existing vector store from disk."""
        from langchain.vectorstores import FAISS
        from langchain.vectorstores.faiss import DistanceStrategy
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store not found at: {path}")
        
        try:
            # Try loading with the latest FAISS API
            self.vector_store = FAISS.load_local(
                folder_path=path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True  # Required for security reasons
            )
        except Exception as e:
            # If that fails, try the older method
            try:
                self.vector_store = FAISS.load_local(
                    folder_path=path,
                    embeddings=self.embeddings
                )
            except Exception as e2:
                logger.error(f"Failed to load vector store: {str(e2)}")
                raise
        
    def save_vector_store(self, path: str) -> None:
        """Save the current vector store to disk."""
        if not self.vector_store:
            raise ValueError("No vector store initialized")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save with the latest FAISS API
            self.vector_store.save_local(
                folder_path=path,
                index_name="index"  # Standard name for FAISS index files
            )
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise
        
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents to the query."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
            
        return self.vector_store.similarity_search(query, k=k)
