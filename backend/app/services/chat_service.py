from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from ..config import settings

class ChatService:
    """Handles chat functionality with conversation memory."""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.conversation_chain = None
        self.chat_history = []
        self.initialize_conversation_chain()
        
    def initialize_conversation_chain(self) -> None:
        """Initialize the conversation chain with memory."""
        # Define the prompt template
        qa_template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Chat History: {chat_history}
        
        Question: {question}
        Answer:"""
        
        QA_PROMPT = PromptTemplate(
            template=qa_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Initialize the language model
        llm = ChatOpenAI(
            model_name=settings.CHAT_MODEL,
            temperature=0.7,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Set up memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Create the conversation chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True
        )
        
    async def process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message and return the AI's response."""
        if not self.conversation_chain:
            self.initialize_conversation_chain()
            
        # Get the AI's response
        response = await self.conversation_chain.acall({
            "question": message,
            "chat_history": self.chat_history
        })
        
        # Update chat history
        self.chat_history.append((message, response["answer"]))
        
        # Format the response
        return {
            "answer": response["answer"],
            "sources": self._format_sources(response.get("source_documents", []))
        }
        
    def _format_sources(self, source_documents: List[Document]) -> List[Dict[str, Any]]:
        """Format source documents for the response."""
        sources = []
        for doc in source_documents:
            source = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source)
        return sources
        
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.chat_history = []
        if self.conversation_chain:
            self.conversation_chain.memory.clear()
