import chromadb
import config
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
import os
import joblib
from build_vectordb import BuildVectorDB


from logging_config import configure_file_logger
logger = configure_file_logger(__name__) 




class AskToVectorDB:
    """Helper to query a Chroma collection using SentenceTransformer embeddings.

    Args:
        collection (chromadb.api.models.Collection): A Chroma collection to query.
        embedding_model (SentenceTransformer): Model used to embed queries.
    """

    def __init__(self, collection: chromadb.api.models.Collection, embedding_model_name: str):
        self.collection = collection
        self.embedding_model = SentenceTransformer(embedding_model_name)
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
    client = chromadb.PersistentClient(path=config.VECTORDB_PATH)
    collection = client.get_collection(name="my_embeddings")

    asker = AskToVectorDB(collection=collection, embedding_model_name=config.EMBEDDING_MODEL_NAME)
    logger.info("Loaded persisted collection from %s", config.VECTORDB_PATH)

    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        response = asker.ask(question, n_results=3)
        for doc in response['documents'][0]:
            print(f"- {doc}")