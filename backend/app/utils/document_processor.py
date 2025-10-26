# document_processor

import logging
from pathlib import Path
from ..config import settings
from ..services.vector_store import DocumentLoader, VectorStoreManager
from ..models.rag_model import RAGModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_initial_documents() -> bool:
    """
    Scans the documents directory, processes supported files, and builds or updates the vector store.
    """
    try:
        doc_dir = Path(settings.DOCUMENTS_DIR)
        vs_path = Path(settings.VECTOR_STORE_PATH)

        logger.info("Starting document processing...")
        logger.info(f"Documents directory: {doc_dir}")
        logger.info(f"Vector store path: {vs_path}")

        # Create necessary directories using pathlib
        doc_dir.mkdir(parents=True, exist_ok=True)
        # The parent of the vector store file/directory needs to exist
        vs_path.parent.mkdir(parents=True, exist_ok=True)

        files = [f for f in doc_dir.iterdir() if f.is_file()]
        if not files:
            logger.warning(f"No files found in documents directory: {doc_dir}")
            return False

        logger.info(f"Found {len(files)} files to potentially process.")

        # Initialize components
        rag_model = RAGModel()
        vector_store_manager = VectorStoreManager(rag_model.embeddings)

        all_chunks = []
        supported_extensions = {'.pdf', '.txt', '.docx', '.md'} # Set for faster lookups
        processed_files_count = 0

        for file_path in files:
            if file_path.suffix.lower() not in supported_extensions:
                logger.warning(f"Skipping unsupported file: {file_path.name}")
                continue

            try:
                logger.info(f"Processing: {file_path.name}")
                documents = DocumentLoader.load_document(str(file_path))
                chunks = DocumentLoader.split_documents(
                    documents,
                    chunk_size=int(settings.CHUNK_SIZE),
                    chunk_overlap=int(settings.CHUNK_OVERLAP)
                )
                all_chunks.extend(chunks)
                processed_files_count += 1
                logger.info(f"  -> Processed {len(chunks)} chunks from {file_path.name}")
            except Exception as e:
                logger.error(f"  -> Error processing {file_path.name}: {e}", exc_info=True)
                continue

        if not all_chunks:
            logger.error("No valid document chunks were generated. Vector store not created.")
            return False

        logger.info(f"Creating/updating vector store with {len(all_chunks)} chunks from {processed_files_count} files...")
        vector_store_manager.create_vector_store(all_chunks)
        vector_store_manager.save_vector_store(str(vs_path))

        logger.info(f"Vector store created/updated successfully at: {vs_path}")
        return True

    except Exception as e:
        logger.critical(f"A critical error occurred during document processing: {e}", exc_info=True)
        return False

def check_vector_store() -> bool:
    """
    Checks if a vector store exists and can be loaded successfully.
    """
    vs_path = Path(settings.VECTOR_STORE_PATH)
    if not vs_path.exists():
        logger.info(f"Vector store not found at {vs_path}.")
        return False

    try:
        logger.info(f"Attempting to load vector store from {vs_path}...")
        rag_model = RAGModel()
        rag_model.load_vector_store(str(vs_path))
        logger.info("Vector store loaded successfully.")
        return True
    except Exception as e:
        # Use the logger instead of print for consistency
        logger.error(f"Error loading vector store from {vs_path}: {e}", exc_info=True)
        return False