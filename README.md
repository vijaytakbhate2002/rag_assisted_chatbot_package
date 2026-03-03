# RAG-Assisted Chatbot — Portfolio & Interview Assistant 💼🤖

**RAG-Assisted Chatbot** is an installable Python package that powers a portfolio-driven HR interview assistant and a companion Medium-article assistant. It uses GitHub README scraping and Medium RSS scraping, PDF conversion, vectorization (ChromaDB), and Retrieval-Augmented Generation (RAG) combined with an LLM to answer HR-style or educational questions based on a candidate's resume, repos or published articles.

---

## 🚀 Highlights

- Scrapes GitHub repositories' README files **and Medium articles** and saves them as well-styled PDFs.
- Builds a persistent Chroma vector store from PDF/HTML content using SentenceTransformers embeddings.
- Queries the vector DB to fetch top-k relevant text chunks for a question (RAG).
- Uses an LLM (e.g., `gpt-5-mini` via `langchain-openai`) with structured output to produce interview-style or educational responses.
- Packaged for easy reuse: import `Assistant`, `BuildVectorDB`, `AskToVectorDB`, `GithubScrapper`, and the Medium collector. Example:

```python
from rag_assisted_bot.rag_assisted_chatbot import Assistant, BuildVectorDB, AskToVectorDB, GithubScrapper
# medium collector is under a separate namespace:
from rag_assisted_bots.ask_medium.data_collection_pipeline import MediumDataCollector
```

(When installed via pip the package name is `rag-assisted-chatbot`.)

For the Medium pipeline the collector lives in `rag_assisted_bots.ask_medium`; import it directly from that module.

---

## Table of contents

- [Quick overview](#-quick-overview)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Quickstart Examples](#-quickstart-examples)
- [How it works (architecture)](#-how-it-works-architecture)
- [Project structure](#-project-structure)
- [Troubleshooting & tips](#-troubleshooting--tips)
- [Contributing & License](#-contributing--license)

---

## 🔍 Quick overview

This project provides four main capabilities:

- **GitHub scraping:** `GithubScrapper` fetches README content and converts to PDF (uses `xhtml2pdf`).
- **Medium scraping:** `MediumDataCollector` parses a user’s Medium RSS feed, styles article HTML and saves each post as a PDF.
- **Vector DB builder:** `BuildVectorDB` loads PDFs, splits text into chunks, encodes with `sentence-transformers`, and stores embeddings in ChromaDB.
- **Assistant:** `Assistant` pairs an LLM with the RAG results to answer HR-style or article‑focused questions using up-to-date context.

---

## 🧩 Installation

Clone and install locally:

```bash
git clone <repo_url>
cd <repo>
python -m pip install -e .
```

Or install dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

> Tip: Use a virtual environment (venv/conda).

---

## ⚙️ Configuration

Project configuration values are in `rag_assisted_bot/rag_assisted_chatbot/config.py`:

- `GITHUB_USERNAME` — username to scrape (default: `"vijaytakbhate2002"`)
- `GITHUB_PDF_FOLDER` — folder to save generated PDFs (default: `"rag_assisted_bot/scrapped_data/github_pdfs"`)
- `METADATA_JSON_PATH` — metadata file location (default: `"scrapped_metadata/metadata.json"`)
- `VECTORDB_PATH` — path for persistent ChromaDB storage (default: `<project_root>/vectordb`)
- `COLLECTION_NAME` — Chroma collection name (default: `"my_embeddings"`)
- `EMBEDDING_MODEL_NAME` — embedding model (default: `"all-MiniLM-L6-v2"`)
- `TOP_K_MATCHES` — number of RAG results to include (default: `3`)
- `GPT_MODEL_NAME` — model used by the assistant (default: `"gpt-5-mini"`)

(Note: the Medium collector does not yet use this config file; paths and username are passed directly when instantiating `MediumDataCollector` or via `data_collection_pipeline_runner.py`.)

Environment variables:

- `TOKEN_GITHUB` — **required** GitHub personal access token (the `GithubScrapper` will raise a `RuntimeError` if not set).

---

## ✨ Quickstart Examples

1. Scrape READMEs and save PDFs

```python
from rag_assisted_bot.rag_assisted_chatbot import GithubScrapper
import rag_assisted_bot.rag_assisted_chatbot.config as cfg

scraper = GithubScrapper(
    username=cfg.GITHUB_USERNAME,
    save_folder=cfg.GITHUB_PDF_FOLDER,
    metadata_save_folder=cfg.METADATA_JSON_PATH
)
scraper.scrap()
```

**Medium example:**

```python
from rag_assisted_bots.ask_medium.data_collection_pipeline import MediumDataCollector

collector = MediumDataCollector("vijaytakbhate45")
collector.save_data(pdf_folder_path="medium_data", metadata_file_path="scrapped_metadata/metadata.json")
```

2. Build the persistent vector database

```python
from rag_assisted_bot.rag_assisted_chatbot import BuildVectorDB
import rag_assisted_bot.rag_assisted_chatbot.config as cfg

builder = BuildVectorDB(
    directory_path=cfg.GITHUB_PDF_FOLDER,
    vectordb_path=cfg.VECTORDB_PATH,
    embedding_model_name=cfg.EMBEDDING_MODEL_NAME,
    collection_name=cfg.COLLECTION_NAME
)
builder.build(chunk_size=300, chunk_overlap=100)
```

3. Run the assistant (interactive programmatic use)

```python
from rag_assisted_bot.rag_assisted_chatbot import Assistant
import rag_assisted_bot.rag_assisted_chatbot.config as cfg

assistant = Assistant(
    gpt_model_name=cfg.GPT_MODEL_NAME,
    temperature=0.7,
    collection_name=cfg.COLLECTION_NAME,
    vectordb_path=cfg.VECTORDB_PATH,
    rag_activated=True
)
result = assistant.chat_with_model("Tell me about your key MLOps achievements")
print(result['question_category'])
print(result['response'].model_dump())
```

4. Query the Vector DB directly

```python
from rag_assisted_bot.rag_assisted_chatbot.ask_vectordb import AskToVectorDB
import chromadb
import rag_assisted_bot.rag_assisted_chatbot.config as cfg

client = chromadb.PersistentClient(path=cfg.VECTORDB_PATH)
collection = client.get_collection(name=cfg.COLLECTION_NAME)
asker = AskToVectorDB(collection=collection, embedding_model_name=cfg.EMBEDDING_MODEL_NAME)
res = asker.ask("Explain your role in the Medical Insurance project", n_results=3)
print(res['documents'][0])
```

---

## 🏗 How it works (architecture)

1. `GithubScrapper` fetches repository README files via GitHub API and saves them as PDFs.
2. `MediumDataCollector` pulls a user’s Medium RSS feed, styles the HTML and also outputs per-article PDFs along with metadata.
3. `BuildVectorDB` loads PDFs with `PyMuPDF`, splits long texts into chunks using `RecursiveCharacterTextSplitter`, and encodes chunks with `SentenceTransformer`.
4. Embeddings are stored in a persistent ChromaDB collection (`my_embeddings`).
5. `AskToVectorDB` (or `GithubAskToVectorDB`) embeds queries and queries ChromaDB for top-k chunks.
6. `Assistant` obtains a category for the question, fetches RAG context (top-k chunks), and calls an LLM with a `SystemMessage` + retrieved context to provide a structured response.

Structured LLM output uses `pydantic` models from `rag_assisted_bot/rag_assisted_chatbot/output_structure.py` (fields include `response_message`, `reference_links`, `confidence_score`, `follow_up_question`).

### Channel comparison

| Channel | Data source | Collector class       | Metadata key | PDF output    | Vector‑store builder    | Assistant mode            |
| ------- | ----------- | --------------------- | ------------ | ------------- | ----------------------- | ------------------------- |
| GitHub  | GitHub API  | `GithubScrapper`      | `github`     | README → PDF  | `GithubBuildVectorDB`   | `assistant_type="github"` |
| Medium  | RSS feed    | `MediumDataCollector` | `medium`     | Article → PDF | `GithubBuildVectorDB`\* | `assistant_type="medium"` |

\*use `metadatas_path` argument or adapt builder to switch on key

---

## 📁 Project structure

Key files and modules:

- `rag_assisted_bot/rag_assisted_chatbot/` — core package modules (GitHub/assistant).
  - `github_scrapper.py` — GitHub README → PDF (method: `scrap()`)
  - `build_vectordb.py` — builds & persists ChromaDB embeddings (`BuildVectorDB`)
  - `ask_vectordb.py` — queries vector DB (`AskToVectorDB`)
  - `main.py` — `Assistant` class and RAG orchestration
  - `prompts.py`, `references.py` — prompts and static resume content
  - `output_structure.py` — pydantic models for structured outputs (`InterViewResponse`, `QuestionCategory`)
  - `config.py` — defaults you can edit (paths & model names)
- `rag_assisted_bots/ask_medium/` — medium scraping pipeline
  - `data_collection_pipeline.py` & runner script
- Output folders created at runtime:
  - `rag_assisted_bot/scrapped_data/` — PDFs (default: `rag_assisted_bot/scrapped_data/github_pdfs`)
  - `medium_data/` or your chosen folder for article PDFs
  - `scrapped_metadata/` — metadata JSON (default: `scrapped_metadata/metadata.json`)
- `vectordb/` (default persistent ChromaDB storage path)

---

## 🔧 Troubleshooting & tips

- Missing GitHub token: `GithubScrapper` will raise `RuntimeError` at import time if `TOKEN_GITHUB` is not set in the environment.
- If embeddings don't persist, ensure `VECTORDB_PATH` is writable and compatible with your `chromadb` version.
- PDF loading requires `PyMuPDF` (package name `PyMuPDF`) and the `langchain_community.document_loaders.PyMuPDFLoader`.
- Check `logs.log` in the project root for runtime logs.

---

## 🤝 Contributing & License

Contributions welcome — open an issue or PR with a clear description of the change. This project is licensed under **MIT**.

---

If you'd like, I can now:

- Add a short `CONTRIBUTING.md` and an example GitHub Actions workflow for CI ✅
- Add a `Makefile` / `scripts/` helpers to run common flows (scrape → build → query) ✅

If any of that sounds useful, tell me which you'd like next!
