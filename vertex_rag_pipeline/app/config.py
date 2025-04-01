import os
import logging
from dotenv import load_dotenv

# Load .env file variables
load_dotenv()

logger = logging.getLogger(__name__)

# --- GCP Settings ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION", "us-central1") # Default to us-central1 if not set

# --- Vertex AI Embeddings Model ---
# See https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-embeddings
# Make sure the model chosen is compatible with your Vector Search index dimensions
VERTEX_EMBEDDING_MODEL = os.getenv("VERTEX_EMBEDDING_MODEL", "textembedding-gecko@003")

# --- Vertex AI Vector Search (Matching Engine) ---
VECTOR_SEARCH_INDEX_ID = os.getenv("VECTOR_SEARCH_INDEX_ID")
VECTOR_SEARCH_INDEX_ENDPOINT_ID = os.getenv("VECTOR_SEARCH_INDEX_ENDPOINT_ID")
# Deployed Index ID is required for querying the index endpoint
VECTOR_SEARCH_DEPLOYED_INDEX_ID = os.getenv("VECTOR_SEARCH_DEPLOYED_INDEX_ID")
# GCS bucket for staging data during indexing (must exist and be in the same region as the index)
GCS_STAGING_BUCKET_NAME = os.getenv("GCS_STAGING_BUCKET_NAME")

# --- Data Loading Settings ---
# List of allowed file extensions for document loading
ALLOWED_SUFFIXES = [".pdf", ".txt", ".md"]
# Whether to search subdirectories recursively
RECURSIVE_LOAD = True

# --- Text Processing Settings ---
# Target number of characters per document chunk
CHUNK_SIZE = 1000
# Number of characters to overlap between consecutive chunks
CHUNK_OVERLAP = 100

# --- Query Settings ---
# Default number of results to retrieve during similarity search
DEFAULT_SEARCH_K = 5

# --- Logging ---
# Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# --- Validation (Optional but Recommended) ---
def validate_config():
    """Perform basic validation of essential configuration."""
    required_vars = [
        "GCP_PROJECT_ID",
        "VECTOR_SEARCH_INDEX_ID",
        "VECTOR_SEARCH_INDEX_ENDPOINT_ID",
        "VECTOR_SEARCH_DEPLOYED_INDEX_ID", # Needed for querying
        "GCS_STAGING_BUCKET_NAME", # Needed for indexing
    ]
    missing_vars = [var for var in required_vars if not globals().get(var)]
    if missing_vars:
        logger.error(f"Missing required configuration variables: {', '.join(missing_vars)}")
        raise ValueError(f"Missing required configuration variables: {', '.join(missing_vars)}. Please check your .env file or environment.")

    if not GCS_STAGING_BUCKET_NAME.startswith('gs://'):
         logger.warning(f"GCS_STAGING_BUCKET_NAME ('{GCS_STAGING_BUCKET_NAME}') does not start with 'gs://'. The Vertex AI client expects the 'gs://' prefix.")
         # Note: The client library might handle this, but explicit check is good.

    logger.info("Configuration loaded and validated (basic checks).")

# You might call validate_config() early in your main.py startup