import logging
from pathlib import Path
from typing import List, Dict, Type, Optional
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader, # Preferred PDF loader
    TextLoader,
    UnstructuredMarkdownLoader,
    BaseLoader, # For type hinting
)
from langchain_core.documents import Document
import sys
import os

# Adjust the Python path to import the config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

# Mapping of file suffixes to LangChain Loader classes
# Ensure required libraries (e.g., pymupdf, unstructured) are installed
DEFAULT_LOADER_MAP: Dict[str, Type[BaseLoader]] = {
    ".pdf": PyMuPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    # Add more mappings here as needed, e.g.:
    # ".csv": CSVLoader,
    # ".docx": UnstructuredWordDocumentLoader,
    # ".html": UnstructuredHTMLLoader,
}

def load_documents(
    source_dir: str,
    allowed_suffixes: Optional[List[str]] = None,
    loader_map: Optional[Dict[str, Type[BaseLoader]]] = None,
    recursive: bool = config.RECURSIVE_LOAD
) -> List[Document]:
    """
    Loads documents from the specified directory using configured loaders,
    filtering by allowed suffixes.

    Args:
        source_dir: The path to the directory containing documents.
        allowed_suffixes: List of file extensions to load (e.g., ['.pdf']).
                          Defaults to config.ALLOWED_SUFFIXES.
        loader_map: Dictionary mapping extensions to loader classes.
                    Defaults to DEFAULT_LOADER_MAP.
        recursive: Whether to search subdirectories. Defaults to config.RECURSIVE_LOAD.


    Returns:
        A list of loaded LangChain Document objects.
    """
    if allowed_suffixes is None:
        allowed_suffixes = config.ALLOWED_SUFFIXES
    if loader_map is None:
        loader_map = DEFAULT_LOADER_MAP

    source_path = Path(source_dir)
    if not source_path.is_dir():
        logger.error(f"Source directory not found: {source_dir}")
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    logger.info(f"Starting document loading from directory: {source_dir}")
    logger.info(f"Allowed file suffixes: {allowed_suffixes}")
    logger.info(f"Recursive loading: {recursive}")

    loaded_docs: List[Document] = []
    loaders_used = set()
    files_processed = 0
    files_failed = 0

    # Use DirectoryLoader for each supported suffix type
    for suffix in allowed_suffixes:
        specific_loader = loader_map.get(suffix.lower()) # Use lower case for matching
        if not specific_loader:
            logger.warning(f"No specific loader configured for suffix '{suffix}'. Skipping files with this extension.")
            continue

        glob_pattern = f"**/*{suffix}" if recursive else f"*{suffix}"
        try:
            # Note: silent_errors=True in DirectoryLoader logs errors but returns successfully loaded docs.
            # Individual file loading errors are handled within the loader itself.
            loader = DirectoryLoader(
                str(source_path), # DirectoryLoader expects string path
                glob=glob_pattern,
                loader_cls=specific_loader,
                recursive=recursive, # Already handled by glob pattern? Redundant but safe.
                show_progress=True, # Show a progress bar during loading
                use_multithreading=True, # Attempt to speed up loading
                silent_errors=True, # Log file loading errors and continue
                # loader_kwargs can be passed here if needed, e.g., {'encoding': 'utf-8'} for TextLoader
            )

            docs_for_suffix = loader.load() # This performs the actual loading

            if docs_for_suffix:
                loaded_docs.extend(docs_for_suffix)
                loaders_used.add(specific_loader.__name__)
                logger.info(f"Loaded {len(docs_for_suffix)} documents with suffix '{suffix}' using {specific_loader.__name__}.")
                # We don't easily know how many failed from DirectoryLoader with silent_errors=True
                # To get precise counts, one might need to glob manually and load file-by-file.
            else:
                 logger.info(f"No documents found or loaded for suffix '{suffix}' in {source_dir} (check glob pattern and file contents).")

            # Estimate processed files (less accurate with silent_errors)
            # files_processed += len(list(source_path.rglob(f"*{suffix}"))) if recursive else len(list(source_path.glob(f"*{suffix}")))


        except ImportError as e:
             logger.error(f"Missing dependency for loader {specific_loader.__name__} required for '{suffix}' files: {e}. Please install it.")
             files_failed += len(list(source_path.rglob(f"*{suffix}"))) if recursive else len(list(source_path.glob(f"*{suffix}"))) # Estimate failures
        except Exception as e:
            # Catch potential errors during DirectoryLoader initialization or broad loading issues
            logger.exception(f"Error loading files with suffix '{suffix}' from {source_dir}: {e}")
            # Count potential failures if error is broad
            files_failed += len(list(source_path.rglob(f"*{suffix}"))) if recursive else len(list(source_path.glob(f"*{suffix}")))

    # A more accurate way to count processed/failed would involve globbing first
    all_files = list(source_path.rglob("*")) if recursive else list(source_path.glob("*"))
    files_processed = len([f for f in all_files if f.is_file() and f.suffix.lower() in allowed_suffixes])


    if not loaded_docs:
        logger.warning(f"No documents were successfully loaded from {source_dir} with the specified criteria.")
    else:
        # Imperfect failure count due to silent_errors=True
        # files_failed = files_processed - len(loaded_docs) # This isn't quite right if one doc fails loading
        logger.info(f"Finished loading. Total documents successfully loaded: {len(loaded_docs)} / ~{files_processed} potential files.")
        # logger.info(f"Estimated files failed or skipped: {files_failed}") # Less reliable count
        logger.info(f"Loaders used: {', '.join(loaders_used)}")

    return loaded_docs