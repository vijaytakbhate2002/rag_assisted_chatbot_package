from pathlib import Path

# GitHub settings
GITHUB_USERNAME = "vijaytakbhate2002"
GITHUB_PDF_FOLDER = "rag_assisted_bot/scrapped_data/github_pdfs"
METADATA_JSON_PATH = "scrapped_metadata/metadata.json"

# Project root as base directory
BASE_DIR = Path(__file__).resolve().parents[1]  

# Chroma vector DB path (writable in project root)
VECTORDB_PATH = str(BASE_DIR / "vectordb")
COLLECTION_NAME = "my_embeddings"

# Embeddings and LLM
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_MATCHES = 4
GPT_MODEL_NAME = "gpt-5-mini"
