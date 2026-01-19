import chromadb
from rag_assisted_bot.rag_assisted_chatbot import config
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

class AskToVectorDB:
    """Helper to query a Chroma collection using SentenceTransformer embeddings.

    Args:
        collection (chromadb.api.models.Collection): A Chroma collection to query.
        embedding_model (SentenceTransformer): Model used to embed queries.
    """

    def __init__(self, collection: chromadb.api.models.Collection, embedding_model_name: str):
        self.collection = collection
        self.embedding_model = SentenceTransformer(embedding_model_name)


    def generate_embeddings(self, query: str) -> list:
        """Generate an embedding vector for the provided query string.

        Args:
            query (str): Query string to be embedded.

        Returns:
            list: The generated embedding vector (as a plain Python list).
        """
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
        result = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )
        return result
    


    def ask(self, query: str, n_results: int = 5):
        """Embed a query and return top relevant chunks from the collection.

        Args:
            query (str): Natural language query to search for.
            n_results (int): Number of top results to return.

        Returns:
            The raw result returned by `find_relevant_chunks`.
        """
        query_embeddings = self.generate_embeddings(query)
        relevant_chunks = self.find_relevant_chunks(
            query_embeddings=query_embeddings,
            n_results=n_results
        )
        return relevant_chunks


