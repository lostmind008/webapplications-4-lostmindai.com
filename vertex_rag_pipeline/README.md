# Vertex AI RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline using Python, LangChain, and Google Cloud Vertex AI services (Embeddings and Vector Search). It allows indexing documents from a local directory into a Vertex AI Vector Search index and querying that index to find relevant document chunks.

## Features

*   Loads documents (`.pdf`, `.txt`, `.md` supported by default) from a specified directory.
*   Splits documents into manageable chunks.
*   Generates text embeddings using Vertex AI `textembedding-gecko` models (configurable).
*   Indexes document chunks into a pre-existing Vertex AI Vector Search (Matching Engine) index.
*   Provides a command-line interface (CLI) for both indexing and querying.
*   Uses Application Default Credentials (ADC) for GCP authentication.
*   Configuration driven via environment variables (`.env` file).

## Prerequisites

1.  **Google Cloud Project:** You need a GCP project with billing enabled.
2.  **Enable APIs:** Ensure the following APIs are enabled in your GCP project:
    *   Vertex AI API (`aiplatform.googleapis.com`)
    *   Cloud Storage API (`storage.googleapis.com`)
3.  **Authentication:** Configure Application Default Credentials (ADC) for your environment. The easiest way locally is to run:
    ```bash
    gcloud auth application-default login
    ```
    Ensure the authenticated principal (user or service account) has necessary IAM roles (e.g., `Vertex AI User`, `Storage Object Admin` on the staging bucket).
4.  **Vertex AI Vector Search Index:** You must **manually create** a Vertex AI Vector Search index **before** running the indexing pipeline.
    *   Choose the correct region.
    *   Configure the index dimensions to match the embedding model output (e.g., 768 for `textembedding-gecko@003`).
    *   Choose an appropriate distance metric (e.g., `DOT_PRODUCT_DISTANCE` or `COSINE_DISTANCE` often work well for text embeddings).
    *   See [Vertex AI Vector Search Docs](https://cloud.google.com/vertex-ai/docs/vector-search/overview) for creation details.
5.  **Vertex AI Index Endpoint:** Create an Index Endpoint in the same region as your index.
6.  **Deploy Index:** Deploy your created index to the Index Endpoint. Note the **Deployed Index ID**.
7.  **GCS Staging Bucket:** Create a Google Cloud Storage bucket in the **same region** as your Vertex AI Index. This is required by the Vector Search API for staging data during indexing. The service account used by Vertex AI needs write access to this bucket.

## Configuration

Configuration is managed via environment variables. Create a `.env` file in the app directory by copying the `.env.example` file:

```bash
cp .env.example .env
```

Now, edit the `.env` file and fill in the required values:

*   `GCP_PROJECT_ID`: Your Google Cloud project ID.
*   `GCP_REGION`: The GCP region where your Vertex AI Index and GCS bucket reside (e.g., `us-central1`).
*   `VERTEX_EMBEDDING_MODEL`: The embedding model to use (ensure compatibility with index dimensions).
*   `VECTOR_SEARCH_INDEX_ID`: The ID of your pre-created Vector Search index.
*   `VECTOR_SEARCH_INDEX_ENDPOINT_ID`: The ID of the Index Endpoint where your index is deployed.
*   `VECTOR_SEARCH_DEPLOYED_INDEX_ID`: The ID of the *specific deployment* of your index on the endpoint (used for querying).
*   `GCS_STAGING_BUCKET_NAME`: The name of your GCS bucket for staging (without `gs://`).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/lostmind008/webapplications-4-lostmindai.com.git
    cd webapplications-4-lostmindai.com/vertex_rag_pipeline
    ```
2.  Create and activate a Python virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The application provides a command-line interface via `main.py`.

### Indexing Documents

To process documents from a directory and add them to your configured Vertex AI Vector Search index, use the `--index` flag along with the `--source` flag specifying the directory path:

```bash
cd app
python main.py --index --source /path/to/your/documents
```

The pipeline will:
1.  Load supported files from the source directory.
2.  Split them into chunks.
3.  Generate embeddings (implicitly during `add_documents`).
4.  Initiate the process of adding them to the Vector Search index via the staging bucket.

*Note:* The actual index update on Vertex AI might take some time after the script finishes.

### Querying Documents

To search the indexed documents for information relevant to a query, use the `--query` flag:

```bash
cd app
python main.py --query "Your question about the documents here"
```

You can specify the number of results to retrieve using the `-k` or `--top_k` flag (defaults to 5):

```bash
python main.py --query "What are the main safety procedures?" -k 10
```

The script will:
1.  Generate an embedding for your query.
2.  Perform a similarity search against the deployed index.
3.  Print the most relevant document chunks found, along with their sources and similarity scores.

## Project Structure

```
vertex_rag_pipeline/
├── app/
│   ├── config.py               # Loads and manages configuration from .env
│   ├── main.py                 # CLI entry point, orchestrates pipelines
│   ├── utils/
│   │   ├── gcp_clients.py      # Initializes Vertex AI clients (Embeddings, Vector Search)
│   │   ├── data_loader.py      # Handles loading documents from files
│   │   ├── text_processing.py  # Handles splitting documents into chunks
│   │   └── vector_store_interface.py # Interacts with Vertex AI Vector Search index
│   └── .env                    # Actual environment variables (create from .env.example)
├── docs/                       # Additional documentation (if needed)
├── requirements.txt            # Python package dependencies
├── README.md                   # This file
└── .env.example                # Example environment variable file
```

## Logging

The application uses Python's standard `logging` module. Log messages are printed to the console. The log level can be configured via the `LOG_LEVEL` environment variable (e.g., `DEBUG`, `INFO`, `WARNING`).

## Contributing

Contributions to improve the pipeline are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## About

This project is developed by [LostMind AI](https://lostmindai.com) to demonstrate integration between LangChain and Google Cloud Vertex AI for building RAG applications.