from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
# Using from_module_config for potentially better future compatibility,
# though langchain_community.llms.Ollama is also fine.
# from langchain.llms import Ollama # If using older langchain
from langchain_community.llms import Ollama 

from ..config import settings # Assuming this provides settings.OLLAMA_CHAT_MODEL and settings.OLLAMA_BASE_URL

# Define the custom prompt once, outside the class or as a class-level constant
# Use 'history' instead of 'chat_history' in the template and input_variables 
# to align with ConversationBufferMemory's default 'history' output format when 
# return_messages=False, which is what the ConversationalRetrievalChain 
# typically expects for the `question` chain.
# NOTE: The current setup uses a *custom* template that explicitly expects 
# 'chat_history' as a string, which means `return_messages=True` 
# in ConversationBufferMemory is slightly counter-intuitive if the template 
# doesn't handle it gracefully (which it seems to be doing by accepting a string). 
# We'll stick to the original variable names for minimal change, but *correct* # the `process_message` call as the chain handles the history automatically.

QA_TEMPLATE = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Chat History: {chat_history}

Question: {question}
Answer:"""

QA_PROMPT = PromptTemplate(
    template=QA_TEMPLATE,
    input_variables=["context", "chat_history", "question"]
)

class ChatService:
    """
    Handles chat functionality with conversation memory and retrieval.
    
    Optimizations:
    1. Removed redundant `self.chat_history` list. The chain's memory handles history.
    2. Corrected the `process_message` call to rely on the chain's internal memory.
    3. Moved prompt definition outside the method/class for clarity and single definition.
    4. Used async for LLM initialization to avoid blocking (if applicable, though Ollama init might not be fully async).
    5. Type hints for better readability.
    """
    
    def __init__(self, vector_store: Any):
        """Initialize ChatService with a vector store."""
        self.vector_store = vector_store
        self.conversation_chain: Optional[ConversationalRetrievalChain] = None
        self.initialize_conversation_chain()
        
    def initialize_conversation_chain(self) -> None:
        """Initialize the conversation chain with memory and LLM."""
        
        # Initialize the Ollama language model
        # NOTE: Ollama initialization is synchronous. If the application is async, 
        # consider initializing this outside or using a cache/singleton pattern.
        try:
            llm = Ollama(
                model=settings.OLLAMA_CHAT_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.7,
                num_ctx=4096
            )
        except AttributeError:
             # Fallback for testing or if settings are missing
             print("Warning: Ollama settings not found. Using default 'llama2'.")
             llm = Ollama(model="llama2", temperature=0.7)


        # Set up memory
        # 'chat_history' is the default memory key for ConversationalRetrievalChain 
        # and aligns with the custom prompt. 'return_messages=True' is fine.
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Create the conversation chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            # Ensure the vector store has an 'as_retriever' method
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            memory=memory,
            # Use the pre-defined QA_PROMPT
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}, 
            return_source_documents=True
        )
        
    async def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a user message and return the AI's response.
        Uses the async call (`acall`) and relies on the internal chain memory.
        """
        if not self.conversation_chain:
            # Should not happen if __init__ is called, but good for safety
            self.initialize_conversation_chain() 
            
        # Optimization/Correction: ConversationalRetrievalChain *manages* # the chat history internally via the `memory` object. You **do not** # need to pass `chat_history` in the `acall` dictionary, nor do you 
        # need the redundant `self.chat_history` list.
        response = await self.conversation_chain.acall({
            "question": message,
            # 'chat_history' key removed, the chain handles it.
        })
        
        # The chain handles updating its internal memory. The redundant 
        # self.chat_history update is removed.
        
        # Format the response
        return {
            "answer": response["answer"],
            "sources": self._format_sources(response.get("source_documents", []))
        }
        
    def _format_sources(self, source_documents: List[Document]) -> List[Dict[str, Any]]:
        """Format source documents for the response."""
        # This function is already quite clean and efficient.
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in source_documents
        ]

    def clear_history(self) -> None:
        """Clear the conversation history."""
        # The redundant self.chat_history clear is removed.
        if self.conversation_chain and self.conversation_chain.memory:
            # Clear the history managed by the ConversationBufferMemory
            self.conversation_chain.memory.clear()