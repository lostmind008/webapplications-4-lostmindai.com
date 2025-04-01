import logging
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_google_vertexai import VertexAIVectorSearch # The vector store class itself
from google.api_core.exceptions import GoogleAPIError, FailedPrecondition
import sys
import os

# Adjust the Python path to import the config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

def add_documents_to_vector_store(vector_store: VertexAIVectorSearch, documents: List[Document], batch_size: int = 500):
    """
    Adds document chunks to the configured Vertex AI Vector Search index.

    Handles batching and potential errors during the upsert process.
    The underlying LangChain implementation handles staging to GCS.

    Args:
        vector_store: An initialized VertexAIVectorSearch client instance.
        documents: A list of LangChain Document objects (chunks) to add.
        batch_size: Number of documents to add in each batch call.
    """
    if not documents:
        logger.warning("No documents provided to add to the vector store. Skipping.")
        return

    total_docs = len(documents)
    logger.info(f"Attempting to add {total_docs} document chunks to Vertex AI Vector Search index: {config.VECTOR_SEARCH_INDEX_ID}...")

    added_count = 0
    try:
        # Add documents in batches to avoid potential API limits or large requests
        for i in range(0, total_docs, batch_size):
            batch = documents[i : i + batch_size]
            logger.info(f"Adding batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size} ({len(batch)} documents)...")

            # The add_documents method handles embedding and GCS staging internally.
            # It returns the document IDs assigned by the vector store.
            ids = vector_store.add_documents(batch) #, request_metadata=[("namespace", "your_rag_namespace")]) # Optional: Add metadata like namespace

            added_count += len(batch) # Assuming success if no exception
            logger.debug(f"Successfully added batch. Example IDs: {ids[:5]}...")

        logger.info(f"Successfully initiated adding {added_count}/{total_docs} documents to the vector store index.")
        logger.info("Note: Index update on Vertex AI may take some time to complete after this operation returns.")

    except FailedPrecondition as e:
         logger.exception(f"FailedPrecondition error while adding documents (often indicates index/endpoint not ready or issues with staging bucket): {e}")
         raise RuntimeError("Failed to add documents due to precondition failure. Check index status, endpoint deployment, and GCS staging bucket permissions/existence.") from e
    except GoogleAPIError as e:
        logger.exception(f"Vertex AI API error while adding documents to index {config.VECTOR_SEARCH_INDEX_ID}. Processed {added_count} documents before error.")
        raise # Re-raise after logging the partial success count
    except Exception as e:
        logger.exception(f"Unexpected error while adding documents. Processed {added_count} documents before error.")
        raise # Re-raise after logging

def query_vector_store(vector_store: VertexAIVectorSearch, query: str, k: int = config.DEFAULT_SEARCH_K) -> List[Tuple[Document, float]]:
    """
    Performs a similarity search against the deployed Vertex AI Vector Search index.

    Args:
        vector_store: The initialized VertexAIVectorSearch client instance.
        query: The user query string.
        k: The number of nearest neighbors to retrieve. Defaults to config.DEFAULT_SEARCH_K.

    Returns:
        A list of tuples, each containing a retrieved Document chunk and its similarity score.
        Returns an empty list if no results are found or an error occurs (after logging).
    """
    if not config.VECTOR_SEARCH_DEPLOYED_INDEX_ID:
        logger.error("VECTOR_SEARCH_DEPLOYED_INDEX_ID is not set in config. Cannot perform query.")
        raise ValueError("Querying requires VECTOR_SEARCH_DEPLOYED_INDEX_ID to be configured.")

    logger.info(f"Performing similarity search for query: '{query[:80]}...'")
    logger.info(f"Targeting deployed index ID: {config.VECTOR_SEARCH_DEPLOYED_INDEX_ID} on endpoint {config.VECTOR_SEARCH_INDEX_ENDPOINT_ID}")
    logger.info(f"Retrieving top k={k} results.")

    results: List[Tuple[Document, float]] = []
    try:
        # similarity_search_with_score embeds the query using the store's embedding function
        # and queries the specified deployed index.
        results = vector_store.similarity_search_with_score(
            query=query,
            k=k,
            # *** CRITICAL: Specify the Deployed Index ID for querying ***
            deployed_index_id=config.VECTOR_SEARCH_DEPLOYED_INDEX_ID,
            # Optional: Add filtering based on metadata stored during indexing
            # filter={"namespace": "your_rag_namespace", "source_year": 2023}
        )
        logger.info(f"Found {len(results)} relevant document chunks.")
        if results:
            logger.debug(f"Top result score: {results[0][1]:.4f}, Source: {results[0][0].metadata.get('source', 'N/A')}")

    except GoogleAPIError as e:
        logger.exception(f"Vertex AI API error during similarity search on deployed index {config.VECTOR_SEARCH_DEPLOYED_INDEX_ID}. Error: {e}")
        # Decide whether to raise or return empty list. Returning empty might be safer for user experience.
        # raise RuntimeError(f"Query failed due to API error: {e}") from e
        print(f"\n[Error] Query failed due to API error: {e}\n", file=sys.stderr) # Also print for CLI visibility
    except Exception as e:
        logger.exception(f"Unexpected error during similarity search.")
        # raise RuntimeError(f"Query failed unexpectedly: {e}") from e
        print(f"\n[Error] Query failed unexpectedly: {e}\n", file=sys.stderr)

    return results # Return empty list on error after logging