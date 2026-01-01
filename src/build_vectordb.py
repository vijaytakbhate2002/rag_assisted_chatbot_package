"""Utilities for building a local Chroma vector store from document files.

This module provides BuildVectorDB which loads documents from a directory, splits
text into chunks, computes embeddings using SentenceTransformers, and stores
embeddings in a Chroma collection.
"""

import logging
import config
import joblib
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from sentence_transformers import SentenceTransformer
import uuid

from logging_config import configure_file_logger
logger = configure_file_logger(__name__) 



class BuildVectorDB:
    """Helper to build a Chroma vector database from documents in a directory.

    Args:
        directory_path (str): Path to the directory containing documents (PDFs supported).
        embedding_model_name (str): SentenceTransformer model name to use for embeddings.
        collection_name (str): Name of the Chroma collection to create/get.
    """


    def __init__(self, directory_path: str, embedding_model_name: str = "all-MiniLM-L6-v2", collection_name: str = "my_embeddings"):
        self.directory_path = directory_path
        
        # Use PersistentClient instead of Client
        # This automatically handles "saving" to the path
        self.client = chromadb.PersistentClient(path=config.VECTORDB_PATH)

        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        logger.info("Initialized Persistent ChromaDB at %s", config.VECTORDB_PATH)


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
        logger.info("Example of a chunk %s", chunks[0].metadata['file_path'].split("/")[-1])
        
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
                
        

if __name__ == "__main__":
    # Example usage
    builder = BuildVectorDB(directory_path=config.GITHUB_PDF_FOLDER)
    builder.build(chunk_size=300, chunk_overlap=100)
