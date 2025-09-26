# RAG Chatbot Backend

A FastAPI-based backend for a RAG (Retrieval-Augmented Generation) chatbot that supports both OpenAI and Ollama models.

## Features

- Support for both OpenAI and Ollama models
- Document processing for various file formats (PDF, TXT, DOCX, MD)
- Vector store for efficient document retrieval
- RESTful API for easy integration with frontend applications
- Configurable chunking and embedding settings

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) (if using Ollama models)
- OpenAI API key (if using OpenAI models)

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd chatbot-rag/backend
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the `backend` directory with your configuration:
   ```env
   # Required for OpenAI
   OPENAI_API_KEY=your_openai_api_key
   
   # Optional: Ollama settings (defaults shown)
   OLLAMA_BASE_URL=http://localhost:11434
   
   # Optional: Model provider (openai or ollama)
   MODEL_PROVIDER=openai
   
   # Optional: Document and vector store paths
   DOCUMENTS_DIR=data/documents
   VECTOR_STORE_PATH=data/vector_store
   ```

## Usage

1. **Start the server**:
   ```bash
   uvicorn app.main:app --reload
   ```

2. **API Endpoints**:

   - `GET /`: Health check
   - `POST /upload/`: Upload a document to the documents directory
   - `POST /initialize/`: Process all documents in the documents directory and create/update the vector store
   - `POST /chat/`: Send a message to the chatbot
   - `POST /switch-model/`: Switch between OpenAI and Ollama models
   - `POST /reset/`: Reset the chat history

3. **Using the API**:

   - **Upload a document**:
     ```bash
     curl -X POST -F "file=@your_document.pdf" http://localhost:8000/upload/
     ```

   - **Initialize the knowledge base**:
     ```bash
     curl -X POST http://localhost:8000/initialize/
     ```

   - **Chat with the bot**:
     ```bash
     curl -X POST -H "Content-Type: application/json" -d '{"message": "Hello, how are you?"}' http://localhost:8000/chat/
     ```

   - **Switch model provider**:
     ```bash
     curl -X POST -H "Content-Type: application/json" -d '{"provider": "ollama"}' http://localhost:8000/switch-model/
     ```

## Document Processing

1. Place your documents in the `data/documents` directory (or the path specified in `.env`)
2. Supported formats: PDF, TXT, DOCX, MD
3. Run the initialization endpoint to process all documents and create the vector store
4. The vector store will be saved to `data/vector_store` (or the path specified in `.env`)

## Model Configuration

### OpenAI
- Set `MODEL_PROVIDER=openai` in `.env`
- Set your `OPENAI_API_KEY`
- Default models:
  - Embedding: `text-embedding-ada-002`
  - Chat: `gpt-3.5-turbo`

### Ollama
1. Install and start Ollama: https://ollama.ai/
2. Pull the required models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama2  # or any other model you prefer
   ```
3. Set `MODEL_PROVIDER=ollama` in `.env`
4. Default models:
   - Embedding: `nomic-embed-text`
   - Chat: `llama2`

## Development

- The API documentation is available at `http://localhost:8000/docs` when the server is running
- Logs are printed to the console
- To run tests (after installing test dependencies):
  ```bash
  pytest
  ```

## License

MIT
