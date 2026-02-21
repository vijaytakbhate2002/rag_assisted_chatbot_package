"""Utilities for building a local Chroma vector store from document files.

This module provides BuildVectorDB which loads documents from a directory, splits
text into chunks, computes embeddings using SentenceTransformers, and stores
embeddings in a Chroma collection.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from sentence_transformers import SentenceTransformer
import uuid
from rag_assisted_bot.rag_assisted_chatbot.config import VECTORDB_PATH
from rag_assisted_bot.rag_assisted_chatbot.logging_config import configure_file_logger
from typing import Union
import json

logger = configure_file_logger(__name__) 


class BuildVectorDB:
    """Helper to build a Chroma vector database from documents in a directory.

    Args:
        directory_path (str): Path to the directory containing documents (PDFs supported).
        embedding_model_name (str): SentenceTransformer model name to use for embeddings.
        collection_name (str): Name of the Chroma collection to create/get.
    """


    def __init__(self, directory_path: str, vectordb_path:str, metadatas_path:str=None,  embedding_model_name: str = "all-MiniLM-L6-v2", collection_name: str = "my_embeddings"):
        self.directory_path = directory_path
        self.metadatas_path = metadatas_path
    
        self.client = chromadb.PersistentClient(path=vectordb_path)
        print("---------------------------vectordb_path---------------------------", vectordb_path)
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        print("------------------self.client.list_collections()-------------------", self.client.list_collections())
        
        logger.info("Initialized Persistent ChromaDB at %s", vectordb_path)
    

    def read_metadata(self) -> Union[list, None]:
        """Read metadata from a JSON file if metadatas_path is set."""
        if not self.metadatas_path:
            logger.warning("No metadatas_path provided; skipping metadata loading.")
            print("No metadatas_path provided; skipping metadata loading.")
            return None

        try:
            with open(self.metadatas_path, 'r') as f:
                metadatas = json.load(f)
            logger.info("Loaded metadata for %d documents from %s", len(metadatas), self.metadatas_path)
            print("Loaded metadata for %d documents from %s", len(metadatas), self.metadatas_path)
            return metadatas['github']
        except Exception as e:
            logger.exception("Failed to read metadata from %s: %s", self.metadatas_path, e)
            print("Failed to read metadata from %s: %s", self.metadatas_path, e)
            return None


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
        document_names = []
        for chunk in split_doc:
            document_names.append(chunk.metadata["source"])
        logger.info("Split into %d chunks", len(split_doc))

        return split_doc, document_names


    def generate_embeddings(self, chunks: list, metadatas:list=None) -> None:
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
                documents=texts,
                metadatas = metadatas
            )
            logger.info("Added %d embeddings to collection '%s'", len(ids), getattr(self.collection, 'name', 'unknown'))
        except Exception as e:
            logger.exception("Failed to generate or add embeddings: %s", e)
            raise


    def build(self, chunks, metadatas) -> list:
        """High-level convenience method to build the vector DB end-to-end.

        Loads documents, splits them into chunks and generates embeddings which are
        stored in the configured Chroma collection.

        Returns:
            List[str]: List of document names corresponding to the chunks added to the collection.
        """
        # Generate and add embeddings to the collection
        self.generate_embeddings(chunks=chunks, metadatas=metadatas)
        logger.info("Finished build for collection '%s'", getattr(self.collection, 'name', 'unknown'))
                
