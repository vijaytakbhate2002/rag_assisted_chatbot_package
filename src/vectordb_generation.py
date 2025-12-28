import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import uuid
from typing import Any, Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class ChromaDB:
    def __init__(self, collection_name: str):
        self.client = chromadb.PersistentClient(Settings(chroma_db_path="./chromadb"))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_document(self, document: Dict[str, Any]):
        self.collection.add(
            documents=[document["content"]],
            metadatas=[document["metadata"]],
            ids=[str(uuid.uuid4())],
        )

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["metadatas", "distances"],
        )
        return list(zip(results["metadatas"][0], results["distances"][0]))