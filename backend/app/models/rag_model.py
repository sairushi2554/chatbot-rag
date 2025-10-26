from typing import List, Dict, Any, Optional, Union
import logging
import os
from pathlib import Path

# --- LangChain Core Imports ---
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings # Use core for base class
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.language_models import BaseChatModel # Could be used for type hints

# --- LangChain Community Imports ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

logger = logging.getLogger(__name__)

# Assuming ..config provides settings and ModelProvider
from ..config import settings, ModelProvider

class RAGModel:
    """Manages the RAG pipeline using LCEL for efficient chaining."""

    def __init__(self):
        # Initialize components once
        self.embeddings: Embeddings = self._get_embeddings()
        self.llm = self._get_llm() # Store the initialized Ollama instance
        self.vector_store: Optional[FAISS] = None
        self.qa_chain: Optional[Runnable] = None
        
    def _get_embeddings(self) -> Embeddings:
        """Initialize MiniLM embeddings from sentence-transformers."""
        logger.info(f"Loading MiniLM embeddings: {settings.EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': settings.DEVICE or 'cpu'}, # Use setting for device
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32 
            }
        )
    
    def _get_llm(self) -> Ollama:
        """Initialize the Ollama language model."""
        logger.info(f"Initializing Ollama LLM with model: {settings.OLLAMA_CHAT_MODEL}")
        # Return the Ollama instance directly
        return Ollama(
            model=settings.OLLAMA_CHAT_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.7,
            num_ctx=4096,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.1,
            # Common stop words for Ollama chat models
            stop=["<|im_end|>", "<|endoftext|>"] 
        )
        
    def create_vector_store(self, documents: List[Document]) -> None:
        """Create a new vector store from documents."""
        if not documents:
            logger.warning("Attempted to create vector store from an empty document list.")
            return
        logger.info(f"Creating vector store from {len(documents)} document(s).")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
    def load_vector_store(self, path: Union[str, Path]) -> None:
        """Load an existing vector store from disk."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Vector store not found at: {path}")

        logger.info(f"Loading vector store from: {path}")
        try:
            self.vector_store = FAISS.load_local(
                folder_path=str(path_obj),
                embeddings=self.embeddings,
                # Setting this flag is generally safer for loading older stores
                allow_dangerous_deserialization=True 
            )
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            raise
        
    def save_vector_store(self, path: Union[str, Path]) -> None:
        """Save the current vector store to disk."""
        if not self.vector_store:
            raise ValueError("No vector store to save")
        
        folder_path = Path(path)
        folder_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving vector store to: {folder_path.resolve()}")
        try:
            self.vector_store.save_local(folder_path=str(folder_path))
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise
            
    def initialize_qa_chain(self) -> None:
        """
        Initializes the LCEL RAG chain.
        The chain is structured to return both the answer and the source documents.
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Load or create a vector store first.")
            
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 1. Define the RAG prompt
        system_template = "You are a helpful AI assistant. Use the following context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."
        human_template = """Context: {context} \n\nQuestion: {question}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
        
        # 2. Define the main RAG chain (Retrieval + Generation)
        rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=lambda x: self._format_docs(x["context"]) # Format documents into a single string
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # 3. Combine retrieval and RAG chain, ensuring context and question are passed correctly
        # The chain starts with the user's question, which is passed through to the final output via 'question'
        # and also used by the retriever to get 'context'.
        self.qa_chain = (
            # Map input to its components: context (retrieved docs) and question (original input)
            {"context": retriever, "question": RunnablePassthrough()}
            # Pass everything to the RAG chain
            | rag_chain_from_docs 
        )

        # 4. Final chain that returns a dictionary with 'answer' and 'sources'
        # The retriever is called once, and its result is stored.
        def get_source_documents(input_dict):
            # The context is the list of Documents returned by the retriever
            return {"answer": self.qa_chain, "sources": retriever}

        # FINAL OPTIMIZED CHAIN:
        # We use a cleaner method by passing the question to both the retriever and the RAG chain
        self.qa_chain = (
            RunnablePassthrough.assign(
                context=retriever
            )
            .assign(
                answer=rag_chain_from_docs,
                sources=lambda x: x["context"] # Accesses the raw list of Documents from the 'context' key
            )
            .pick("answer", "sources") # Selects only 'answer' and 'sources' keys for the final output
        )
        
    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """Formats a list of documents into a single string for the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    async def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG model with a question, returning the answer and sources."""
        if not self.qa_chain:
            try:
                self.initialize_qa_chain()
            except ValueError as e:
                logger.error(f"Cannot initialize RAG chain: {e}")
                return {"answer": "Error: RAG model not properly initialized.", "sources": []}
            
        try:
            # The optimized LCEL chain returns a dictionary {"answer": str, "sources": List[Document]}
            result = await self.qa_chain.ainvoke({"question": question})
            
            # Format the sources list from Document objects
            formatted_sources = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } 
                for doc in result["sources"]
            ]
            
            return {
                "answer": result["answer"],
                "sources": formatted_sources
            }
        except Exception as e:
            logger.error(f"Error in query: {str(e)}", exc_info=True)
            return {
                "answer": "Sorry, I encountered an error while processing your question.",
                "sources": []
            }