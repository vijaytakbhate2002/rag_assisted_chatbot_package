from rag_assisted_bots.ask_github.conversation_management import conversationUpdate
from rag_assisted_bots.ask_github.output_structure import InterViewResponse, RagActivation
from rag_assisted_bots.ask_github.prompts import rag_activation_prompt
from rag_assisted_bots.ask_github.ask_vectordb import GithubAskToVectorDB
from rag_assisted_bots.ask_github.config import TOP_K_MATCHES, EMBEDDING_MODEL_NAME
import chromadb
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


class RAGModel:
    """ This is VectorDB communicator which takes question as input and returns the relevant chunks from the VectorDB. """
    def __init__(self, vectordb_path:str, collection_name:str, embedding_model_name:str):
        self.vectordb_path = vectordb_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name


    def build_config(self):
        """
        Build ChromaDB PersistentClient and get the collection. Plus initialize AskToVectorDB 
        Returns:
            None
        """
        client = chromadb.PersistentClient(path=self.vectordb_path)
        self.collection = client.get_collection(name=self.collection_name)
        self.asker = GithubAskToVectorDB(collection=self.collection, embedding_model_name=self.embedding_model_name)

    
    def ask(self, question, n_results) -> str:
        """ This is helper function to ask question to asked """
        response = self.asker.ask(question, n_results=n_results) 
        documents = response['documents']   
        metadatas = response['metadatas']
        return documents, metadatas



class GithubAssistant:
    """ This is an LLM gpt-5-min whcih uses RAG plus resume context to answer interview questions asked by HR."""

    updated_conversation = []


    def __init__(self, gpt_model_name:str, temperature:float, collection_name:str, vectordb_path:str, rag_activated:bool):
        self.gpt_model_name = gpt_model_name
        self.temperature = temperature
        self.rag_activated = rag_activated
        if self.rag_activated:
            self.rag_model = RAGModel(
                                    vectordb_path=vectordb_path,
                                    collection_name=collection_name,
                                    embedding_model_name=EMBEDDING_MODEL_NAME
                                    )
            self.rag_model.build_config()

        if self.gpt_model_name:
            self.model = ChatOpenAI(
                                    model_name=self.gpt_model_name, 
                                    temperature=self.temperature
                                    )
            

    def RAG_context_fetcher(self, question:str, n_results:int) -> str:
        """ This function fetches context from RAG model based on question asked."""
        documents, metadatas = self.rag_model.ask(question, n_results=n_results)
        context = "\n".join([doc for doc in documents[0]])
        return context, metadatas

    def build_chains(self, rag_activation_prompt):
        """ This function will initialize question category model and conversational model. """
        conversational_model = self.model.with_structured_output(InterViewResponse)
        rag_activation_model = self.model.with_structured_output(RagActivation)
        rag_activation_model = rag_activation_prompt | rag_activation_model

        return conversational_model, rag_activation_model

    def remove_duplicates(self, lis:list):
        """Builds unique metadata list of dictionaries by refering repo_name from dictionary"""
        if len(lis) <= 0:
            return []
        unique_metadatas = []
        unique_repo_names = []
        for metadata in lis:
            name = metadata.get("repo_name")
            if name.strip() not in unique_repo_names:
                unique_repo_names.append(name)
                unique_metadatas.append(metadata)
        return unique_metadatas


    def chat_with_model(self, question:str) -> dict:
        """ Takes input question and return answer of that question with updating conversation list.
            conversation list used to make model remember last 4 conversation messages
            Args:
                question: input question

            Returns:
                    (answer, question_category): returns the response of llm """

        conversation_model, rag_activation_chain = self.build_chains(rag_activation_prompt)

        rag_context, metadatas = self.RAG_context_fetcher(
                                                    question=question,
                                                    n_results=TOP_K_MATCHES
                                                    ) if self.rag_activated else ("", [])
        
        rag_activation = rag_activation_chain.invoke({"question": question, "rag_context": rag_context})

        self.updated_conversation = conversationUpdate(
                                                rag_context=rag_context,
                                                top_k_matches=TOP_K_MATCHES,
                                                rag_activation = rag_activation.rag_activation
                                                    )

        self.updated_conversation.append(HumanMessage(question))
        response = conversation_model.invoke(self.updated_conversation)
        self.updated_conversation.append(AIMessage(response.response_message))

        unique_metadatas = self.remove_duplicates(metadatas[0])
        
        return {
                "response":  response,
                "rag_relevance": rag_activation.rag_activation,
                "metadatas": unique_metadatas,
                "rag_context": rag_context
        }

