import logging
from typing import Optional
from google.api_core.exceptions import GoogleAPIError, NotFound
from langchain_google_vertexai import VertexAIEmbeddings, VertexAIVectorSearch
import sys
import os

# Adjust the Python path to import the config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

# Simple in-memory cache for clients
_vertex_embeddings_client: Optional[VertexAIEmbeddings] = None
_vertex_vector_search_client: Optional[VertexAIVectorSearch] = None

def get_vertex_embeddings_client() -> VertexAIEmbeddings:
    """
    Initializes and returns a cached Vertex AI Embeddings client.
    Relies on Application Default Credentials (ADC).
    """
    global _vertex_embeddings_client
    if _vertex_embeddings_client is None:
        if not config.GCP_PROJECT_ID:
             raise ValueError("GCP_PROJECT_ID must be set in config.")
        try:
            logger.info(f"Initializing VertexAIEmbeddings (Project: {config.GCP_PROJECT_ID}, Region: {config.GCP_REGION}, Model: {config.VERTEX_EMBEDDING_MODEL})")
            _vertex_embeddings_client = VertexAIEmbeddings(
                project=config.GCP_PROJECT_ID,
                location=config.GCP_REGION,
                model_name=config.VERTEX_EMBEDDING_MODEL,
                # credentials=None # Explicitly rely on ADC
            )
            # Perform a small test embedding to check connectivity/authentication early
            _vertex_embeddings_client.embed_query("test")
            logger.info("VertexAIEmbeddings client initialized and tested successfully.")
        except GoogleAPIError as e:
            logger.exception(f"Failed to initialize VertexAIEmbeddings client due to Google API error: {e}")
            raise RuntimeError("Could not connect to Vertex AI Embeddings API. Check credentials (ADC), permissions, and API enablement.") from e
        except Exception as e: # Catch other potential issues (e.g., config validation if added)
             logger.exception(f"An unexpected error occurred during VertexAIEmbeddings initialization: {e}")
             raise RuntimeError("Unexpected error initializing Vertex AI Embeddings.") from e
    return _vertex_embeddings_client

def get_vertex_vector_search_client() -> VertexAIVectorSearch:
    """
    Initializes and returns a cached Vertex AI Vector Search client.
    Requires a valid embeddings client. Relies on ADC.
    """
    global _vertex_vector_search_client
    if _vertex_vector_search_client is None:
        if not all([config.GCP_PROJECT_ID, config.VECTOR_SEARCH_INDEX_ID, config.VECTOR_SEARCH_INDEX_ENDPOINT_ID, config.GCS_STAGING_BUCKET_NAME]):
             raise ValueError("Missing required Vector Search config (Project ID, Index ID, Endpoint ID, Staging Bucket Name).")

        # Ensure embeddings client is initialized first
        embeddings_client = get_vertex_embeddings_client() # Can raise error if fails

        # Ensure staging bucket name includes gs:// prefix if not already present
        staging_bucket = config.GCS_STAGING_BUCKET_NAME
        if not staging_bucket.startswith("gs://"):
            staging_bucket = f"gs://{staging_bucket}"
            logger.warning(f"Prepended 'gs://' to GCS_STAGING_BUCKET_NAME. Using: {staging_bucket}")


        try:
            logger.info(f"Initializing VertexAIVectorSearch (Project: {config.GCP_PROJECT_ID}, Region: {config.GCP_REGION}, Index: {config.VECTOR_SEARCH_INDEX_ID}, Endpoint: {config.VECTOR_SEARCH_INDEX_ENDPOINT_ID}, Staging Bucket: {staging_bucket})")
            _vertex_vector_search_client = VertexAIVectorSearch(
                project_id=config.GCP_PROJECT_ID,
                location=config.GCP_REGION,
                index_id=config.VECTOR_SEARCH_INDEX_ID,
                endpoint_id=config.VECTOR_SEARCH_INDEX_ENDPOINT_ID, # Endpoint ID for the index
                embedding=embeddings_client, # Pass the initialized embedding client
                # credentials=None, # Explicitly rely on ADC
                staging_bucket=staging_bucket,
            )
            # Test connection by trying to fetch index details (optional check)
            # Note: The LangChain wrapper doesn't directly expose index/endpoint get methods easily.
            # A low-level client check could be added if needed, but relying on first operation is often sufficient.
            logger.info("VertexAIVectorSearch client initialized successfully.")
        except NotFound as e:
            logger.exception(f"Failed to initialize VertexAIVectorSearch: Resource not found (Index or Endpoint?). Check IDs: {e}")
            raise RuntimeError(f"Vertex AI Vector Search resource not found. Check Index ID '{config.VECTOR_SEARCH_INDEX_ID}' and Endpoint ID '{config.VECTOR_SEARCH_INDEX_ENDPOINT_ID}'.") from e
        except GoogleAPIError as e:
            logger.exception(f"Failed to initialize VertexAIVectorSearch client due to Google API error: {e}")
            raise RuntimeError("Could not connect to Vertex AI Vector Search API. Check credentials (ADC), permissions, API enablement, and resource IDs.") from e
        except Exception as e:
             logger.exception(f"An unexpected error occurred during VertexAIVectorSearch initialization: {e}")
             raise RuntimeError("Unexpected error initializing Vertex AI Vector Search.") from e
    return _vertex_vector_search_client

# Example of how to reset clients if needed (e.g., for testing)
def reset_clients():
    global _vertex_embeddings_client, _vertex_vector_search_client
    _vertex_embeddings_client = None
    _vertex_vector_search_client = None
    logger.debug("GCP clients cache reset.")