import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import sys
import os

# Adjust the Python path to import the config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits loaded documents into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        documents: A list of LangChain Document objects loaded by the data loader.

    Returns:
        A list of smaller Document chunks, preserving metadata from original docs.
    """
    if not documents:
        logger.warning("No documents received for splitting.")
        return []

    logger.info(f"Starting text splitting for {len(documents)} documents...")
    logger.info(f"Using RecursiveCharacterTextSplitter with chunk_size={config.CHUNK_SIZE}, chunk_overlap={config.CHUNK_OVERLAP}")

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,         # Function to measure chunk size (standard character length)
        is_separator_regex=False,    # Treat separators as literal strings
        add_start_index=True,        # Add metadata indicating chunk's start position in original doc
    )

    chunks = []
    doc_count = len(documents)
    try:
        chunks = text_splitter.split_documents(documents) # This does the work
        logger.info(f"Successfully split {doc_count} documents into {len(chunks)} chunks.")

        if chunks:
             # Log details of the first chunk as a sample check
             first_chunk = chunks[0]
             logger.debug(f"Example chunk 1 metadata: {first_chunk.metadata}")
             logger.debug(f"Example chunk 1 content preview: {first_chunk.page_content[:100]}...")
        else:
             logger.warning(f"Splitting resulted in zero chunks for {doc_count} input documents. Check document content and splitter settings.")

        return chunks
    except Exception as e:
        logger.exception(f"An error occurred during document splitting for {doc_count} documents.")
        # Decide on behavior: raise error, return partial results, return empty list?
        # Raising error is often safest for a batch process.
        raise RuntimeError(f"Failed to split documents. Error: {e}") from e