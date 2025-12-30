"""Utilities for building a local Chroma vector store from document files.

This module provides BuildVectorDB which loads documents from a directory, splits
text into chunks, computes embeddings using SentenceTransformers, and stores
embeddings in a Chroma collection.
"""

import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from sentence_transformers import SentenceTransformer
import uuid

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)



class BuildVectorDB:
    """Helper to build a Chroma vector database from documents in a directory.

    Args:
        directory_path (str): Path to the directory containing documents (PDFs supported).
        embedding_model_name (str): SentenceTransformer model name to use for embeddings.
        collection_name (str): Name of the Chroma collection to create/get.
    """

    def __init__(self, directory_path: str, embedding_model_name: str = "all-MiniLM-L6-v2", collection_name: str = "my_embeddings"):
        """Construct the BuildVectorDB.

        Initializes the Chroma client, creates or gets the specified collection and
        loads the SentenceTransformer embedding model.
        """
        self.directory_path = directory_path
        self.client = chromadb.Client(
            Settings(
                persist_directory="./chroma_db",
                anonymized_telemetry=False
            )
        )

        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info("Initialized BuildVectorDB(directory_path=%s, collection=%s, embedding_model=%s)",
                    self.directory_path, collection_name, embedding_model_name)


    def load_documents(self):
        """Load documents from the configured directory using PyMuPDF.

        Returns:
            List[Document]: A list of loaded documents.
        """
        logger.info("Loading documents from directory: %s", self.directory_path)
        dir_pdf_loader = DirectoryLoader(
            self.directory_path,
            loader_cls=PyMuPDFLoader,
            show_progress=True
        )

        dir_content = dir_pdf_loader.load()
        logger.info("Loaded %d documents from %s", len(dir_content), self.directory_path)
        return dir_content

    
    def split_documents(self, documents, chunk_size=200, chunk_overlap=200):
        """Split documents into smaller chunks for embedding.

        Args:
            documents (Sequence[Document]): Documents to split.
            chunk_size (int): Target chunk size in characters.
            chunk_overlap (int): Number of overlapping characters between chunks.

        Returns:
            List[Document]: The list of document chunks.
        """
        logger.info("Splitting %d documents into chunks (chunk_size=%d, chunk_overlap=%d)", len(documents), chunk_size, chunk_overlap)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len, 
            separators=['\n\n', "  "]
        )

        split_doc = splitter.split_documents(documents=documents)
        logger.info("Split into %d chunks", len(split_doc))

        return split_doc


    def generate_embeddings(self, chunks: list) -> None:
        """Generate embeddings for document chunks and add them to the Chroma collection.

        This method encodes document chunk text in batches for efficiency, converts
        embeddings to plain Python lists if necessary, and adds them to the
        configured Chroma collection.

        Args:
            chunks (List[Document]): Document chunks whose `page_content` will be embedded.

        Returns:
            None
        """
        if not chunks:
            logger.warning("No chunks provided to generate_embeddings; skipping.")
            return

        logger.info("Generating embeddings for %d chunks", len(chunks))
        try:
            texts = [str(chunk.page_content) for chunk in chunks]
            ids = [str(uuid.uuid4()) for _ in chunks]

            # Encode in batch for efficiency
            embeddings = self.embedding_model.encode(texts)

            # Ensure embeddings are regular lists for Chroma (e.g., convert numpy arrays)
            try:
                embeddings = embeddings.tolist()
            except Exception:
                embeddings = [list(vec) for vec in embeddings]

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts
            )
            logger.info("Added %d embeddings to collection '%s'", len(ids), getattr(self.collection, 'name', 'unknown'))
        except Exception as e:
            logger.exception("Failed to generate or add embeddings: %s", e)
            raise


    def build(self, chunk_size: int = 200, chunk_overlap: int = 200) -> None:
        """High-level convenience method to build the vector DB end-to-end.

        Loads documents, splits them into chunks and generates embeddings which are
        stored in the configured Chroma collection.
        """
        logger.info("Starting build (chunk_size=%d, chunk_overlap=%d)", chunk_size, chunk_overlap)
        documents = self.load_documents()
        chunks = self.split_documents(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Generate and add embeddings to the collection
        self.generate_embeddings(chunks=chunks)
        logger.info("Finished build for collection '%s'", getattr(self.collection, 'name', 'unknown'))
                



class AskToVectorDB:
    """Helper to query a Chroma collection using SentenceTransformer embeddings.

    Args:
        collection (chromadb.api.models.Collection): A Chroma collection to query.
        embedding_model (SentenceTransformer): Model used to embed queries.
    """

    def __init__(self, collection: chromadb.api.models.Collection, embedding_model: SentenceTransformer):
        self.collection = collection
        self.embedding_model = embedding_model
        logger.info("AskToVectorDB initialized for collection '%s'", getattr(self.collection, 'name', 'unknown'))

    def generate_embeddings(self, query: str) -> list:
        """Generate an embedding vector for the provided query string.

        Args:
            query (str): Query string to be embedded.

        Returns:
            list: The generated embedding vector (as a plain Python list).
        """
        logger.debug("Generating embedding for query: %s", query)
        emb = self.embedding_model.encode([query])
        try:
            return emb.tolist()
        except Exception:
            return [list(e) for e in emb]

    def find_relevant_chunks(self, query_embeddings: list, n_results: int = 5) -> list:
        """Query the collection using pre-computed embeddings and return results.

        Args:
            query_embeddings (list): Embedding vector(s) to query with.
            n_results (int): Number of top results to return.

        Returns:
            list|dict: The raw result returned by the Chroma collection's query method.
        """
        logger.info("Querying collection for top %d results", n_results)
        result = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )
        logger.debug("Query returned result of type %s", type(result))
        return result

    def ask(self, query: str, n_results: int = 5):
        """Embed a query and return top relevant chunks from the collection.

        Args:
            query (str): Natural language query to search for.
            n_results (int): Number of top results to return.

        Returns:
            The raw result returned by `find_relevant_chunks`.
        """
        logger.info("Asking vector DB for query: %s", query)
        query_embeddings = self.generate_embeddings(query)
        relevant_chunks = self.find_relevant_chunks(
            query_embeddings=query_embeddings,
            n_results=n_results
        )
        logger.debug("Relevant chunks: %s", str(relevant_chunks)[:200])
        return relevant_chunks


if __name__ == "__main__":
    # Example usage
    import config
    builder = BuildVectorDB(directory_path=config.GITHUB_PDF_FOLDER)
    builder.build(chunk_size=300, chunk_overlap=100)

    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break    
        asker = AskToVectorDB(
            collection=builder.collection,
            embedding_model=builder.embedding_model
        )
        response = asker.ask(question, n_results=3)
        for doc in response['documents'][0]:
            print(f"- {doc}")