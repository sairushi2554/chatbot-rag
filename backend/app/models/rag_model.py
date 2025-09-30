from typing import List, Dict, Any, Optional, Union
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
import logging

logger = logging.getLogger(__name__)

from ..config import settings, ModelProvider

class RAGModel:
    def __init__(self):
        self.embeddings = self._get_embeddings()
        self.vector_store = None
        self.qa_chain = None
        
    def _get_embeddings(self):
        """Initialize MiniLM embeddings from sentence-transformers."""
        logger.info(f"Loading MiniLM embeddings: {settings.EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have a GPU
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32  # Process embeddings in batches for better performance
            }
        )
    
    def _get_llm(self):
        """Initialize the Ollama language model."""
        logger.info(f"Initializing Ollama LLM with model: {settings.OLLAMA_CHAT_MODEL}")
        return ChatOllama(
            model=settings.OLLAMA_CHAT_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.7,  # Slightly more creative responses
            num_ctx=4096,     # Larger context window
            num_predict=512,  # Maximum number of tokens to predict
            top_k=40,         # Top-k sampling
            top_p=0.9,        # Nucleus sampling
            repeat_penalty=1.1,  # Penalize repetition
            stop=["<|im_end|>", "<|endoftext|>"]  # Common stop sequences
        )
        
    def create_vector_store(self, documents: List[Document]) -> None:
        """Create a new vector store from documents."""
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
    def load_vector_store(self, path: str) -> None:
        """Load an existing vector store from disk."""
        try:
            # Simplified loading method
            self.vector_store = FAISS.load_local(
                folder_path=path,
                embeddings=self.embeddings
            )
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            raise
        
    def save_vector_store(self, path: str) -> None:
        """Save the current vector store to disk."""
        if not self.vector_store:
            raise ValueError("No vector store to save")
            
        try:
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Simplified save operation
            self.vector_store.save_local(folder_path=path)
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise
            
    def initialize_qa_chain(self) -> None:
        """Initialize the QA chain with the current vector store."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Load or create a vector store first.")
            
        # Define the prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Initialize the QA chain with updated parameters
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self._get_llm(),
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG model with a question."""
        if not self.qa_chain:
            self.initialize_qa_chain()
            
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": result.get("source_documents", [])
        }
